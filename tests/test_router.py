"""
Unit tests for the IDOP 5-path semantic query router.

Tests all five classification paths (SQL, MUTATION, RAG, CHAT, HYBRID)
using mocked OpenAI API responses, plus fallback behavior on API failure.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.core.graph.router import QueryRouter, RouteDecision


# ═══════════════════════════════════════════════════════════════════════
# Helper — Mock OpenAI Response Factory
# ═══════════════════════════════════════════════════════════════════════

def _mock_openai_response(query_type: str, reason: str = "Test classification"):
    """Build a mock OpenAI ChatCompletion response returning JSON."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "query_type": query_type,
        "reason": reason,
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ═══════════════════════════════════════════════════════════════════════
# Tests for QueryRouter
# ═══════════════════════════════════════════════════════════════════════

class TestQueryRouter:
    """Tests for the 5-class LLM semantic router."""

    @pytest.fixture
    def router(self):
        """Create a QueryRouter instance with mocked settings."""
        with patch("app.core.graph.router.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            r = QueryRouter()
            r.client = mock_client
            yield r

    # ----- Classification Tests -----

    def test_classify_sql_query(self, router):
        """Test that analytical database questions are classified as SQL."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "SQL", "The user wants to query database tables for product information."
        )
        result = router.route_query("Which products have never been ordered?")
        assert result == "SQL"

    def test_classify_mutation_query(self, router):
        """Test that data modification requests are classified as MUTATION."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "MUTATION", "The user wants to insert new rows from a file."
        )
        result = router.route_query("Insert new products from this Excel file")
        assert result == "MUTATION"

    def test_classify_rag_query(self, router):
        """Test that knowledge search requests are classified as RAG."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "RAG", "The user is asking about a policy document."
        )
        result = router.route_query("What is our company's refund policy?")
        assert result == "RAG"

    def test_classify_chat_query(self, router):
        """Test that general conversational queries are classified as CHAT."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "CHAT", "The user is greeting the system."
        )
        result = router.route_query("Hello, how are you?")
        assert result == "CHAT"

    def test_classify_hybrid_query(self, router):
        """Test that combined SQL + document queries are classified as HYBRID."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "HYBRID", "The user wants database data compared against document guidelines."
        )
        result = router.route_query(
            "Get sales data for customer X and compare it against the sales strategy in our PDF guidelines."
        )
        assert result == "HYBRID"

    # ----- Edge Cases -----

    def test_invalid_query_type_falls_back_to_chat(self, router):
        """Test that an unrecognized query_type from the LLM falls back to CHAT."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "UNKNOWN_TYPE", "Unrecognized classification."
        )
        result = router.route_query("Something weird and unclassifiable")
        assert result == "CHAT"

    def test_api_failure_falls_back_to_chat(self, router):
        """Test that an OpenAI API error causes graceful fallback to CHAT."""
        router.client.chat.completions.create.side_effect = Exception(
            "API rate limit exceeded"
        )
        result = router.route_query("Show total revenue by region")
        assert result == "CHAT"

    def test_malformed_json_falls_back_to_chat(self, router):
        """Test that malformed JSON from the LLM falls back to CHAT."""
        mock_choice = MagicMock()
        mock_choice.message.content = "This is not valid JSON"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        router.client.chat.completions.create.return_value = mock_response
        result = router.route_query("Some query")
        assert result == "CHAT"

    def test_lowercase_query_type_normalized_to_uppercase(self, router):
        """Test that lowercase query_type from LLM is normalized to uppercase."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "sql", "Lowercase classification."
        )
        result = router.route_query("Show me all customers")
        assert result == "SQL"

    def test_mixed_case_query_type_normalized(self, router):
        """Test that mixed-case query_type is normalized correctly."""
        router.client.chat.completions.create.return_value = _mock_openai_response(
            "Hybrid", "Mixed case classification."
        )
        result = router.route_query("Compare data with docs")
        assert result == "HYBRID"

    # ----- Routing Function Tests -----

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
