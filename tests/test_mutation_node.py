"""
Unit tests for the mutation_node — improved message and routing behavior.

Tests verify:
  1. mutation_node returns the correct status and 3-step API instructions
  2. The message includes all three API endpoints (upload, preview, approve)
  3. The node logs as expected
  4. Graceful handling of partial/empty state
  5. route_after_router correctly routes MUTATION to "chat"
"""

import pytest
from unittest.mock import patch

from app.core.graph.nodes import mutation_node, route_after_router

# ═══════════════════════════════════════════════════════════════════════
# mutation_node Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMutationNode:
    """Tests for the mutation_node graph node function."""

    @pytest.mark.asyncio
    async def test_returns_requires_file_upload_status(self):
        """The returned dict must have mutation_status='requires_file_upload'."""
        result = await mutation_node({})
        assert result["mutation_status"] == "requires_file_upload"

    @pytest.mark.asyncio
    async def test_returns_error_message_with_three_api_steps(self):
        """The mutation_error must contain all 3 API endpoint instructions."""
        result = await mutation_node({})
        msg = result["mutation_error"]

        assert "POST /mutation/upload" in msg
        assert "GET /mutation/pending" in msg
        assert "POST /mutation/approve" in msg

    @pytest.mark.asyncio
    async def test_error_message_mentions_csv_or_excel(self):
        """The mutation_error should tell the user to upload a CSV or Excel file."""
        result = await mutation_node({})
        assert "CSV" in result["mutation_error"] or "Excel" in result["mutation_error"]

    @pytest.mark.asyncio
    async def test_logs_info_on_trigger(self):
        """The node should log an info message when triggered."""
        with patch("app.core.graph.nodes.logger") as mock_logger:
            await mutation_node({})
            mock_logger.info.assert_called_once_with(
                "Feature 2 Mutation Node triggered"
            )

    @pytest.mark.asyncio
    async def test_handles_state_with_question(self):
        """mutation_node should work even when state contains a question."""
        result = await mutation_node({"question": "Insert new employees"})
        assert result["mutation_status"] == "requires_file_upload"

    @pytest.mark.asyncio
    async def test_handles_full_state_gracefully(self):
        """mutation_node should not crash when given a full-like state dict."""
        state = {
            "question": "Update product prices",
            "messages": [],
            "query_type": "MUTATION",
            "user_id": "test-user",
        }
        result = await mutation_node(state)
        assert result["mutation_status"] == "requires_file_upload"
        assert "/mutation/upload" in result["mutation_error"]

    @pytest.mark.asyncio
    async def test_returns_only_expected_keys(self):
        """The returned dict should only contain mutation_status and mutation_error."""
        result = await mutation_node({})
        expected_keys = {"mutation_status", "mutation_error"}
        assert set(result.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════
# route_after_router — MUTATION Routing Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMutationRouting:
    """Tests that MUTATION queries route to 'chat' instead of a stub node."""

    def test_mutation_routes_to_chat(self):
        """MUTATION should route to 'chat' so the LLM explains the upload workflow."""
        state = {"query_type": "MUTATION"}
        assert route_after_router(state) == "chat"

    def test_mutation_with_no_state_falls_back_to_chat(self):
        """If query_type is missing but defaults to MUTATION-like, check fallback."""
        state = {}
        # With empty state, route_after_router default is "chat"
        assert route_after_router(state) == "chat"

    def test_mutation_does_not_route_to_unknown_node(self):
        """Ensure MUTATION does NOT route to a non-existent 'mutation' node."""
        state = {"query_type": "MUTATION"}
        result = route_after_router(state)
        assert result != "mutation"  # no such node in the graph builder
        assert result == "chat"

    def test_mutation_routing_always_chat_no_matter_extra_state(self):
        """Extra state fields should not affect MUTATION routing."""
        state = {
            "query_type": "MUTATION",
            "question": "Bulk upload employees via CSV",
            "user_id": "admin",
            "mutation_op": "INSERT",
        }
        assert route_after_router(state) == "chat"

    def test_sql_does_not_route_to_chat(self):
        """SQL queries should still route to 'sql_gen', not 'chat'."""
        state = {"query_type": "SQL"}
        assert route_after_router(state) == "sql_gen"
        assert route_after_router(state) != "chat"

    def test_rag_does_not_route_to_chat(self):
        """RAG queries should still route to 'ltm_remember', not 'chat'."""
        state = {"query_type": "RAG"}
        assert route_after_router(state) == "ltm_remember"
        assert route_after_router(state) != "chat"

    def test_hybrid_does_not_route_to_chat(self):
        """HYBRID queries should route to 'hybrid', not 'chat'."""
        state = {"query_type": "HYBRID"}
        assert route_after_router(state) == "hybrid"
        assert route_after_router(state) != "chat"
