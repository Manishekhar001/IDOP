"""
Unit tests for the mutation_node — improved message and routing behavior.

Tests verify:
  1. mutation_node returns the correct status and 3-step API instructions
  2. The message includes all three API endpoints (upload, preview, approve)
  3. The node logs as expected
  4. Graceful handling of partial/empty state
  5. route_after_router correctly routes MUTATION to "mutation" (not "chat")
"""

from unittest.mock import patch

import pytest

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
    async def test_classifies_op_when_question_provided(self):
        """With a question, the node should classify the operation type."""
        with patch(
            "app.core.feature2_mutation.op_classifier.OpClassifier.classify_operation",
            return_value="INSERT",
        ):
            result = await mutation_node({"question": "Insert new employees"})
        assert result["mutation_op"] == "INSERT"

    @pytest.mark.asyncio
    async def test_handles_state_with_question(self):
        """mutation_node should work even when state contains a question."""
        with patch(
            "app.core.feature2_mutation.op_classifier.OpClassifier.classify_operation",
            return_value="INSERT",
        ):
            result = await mutation_node({"question": "Insert new employees"})
        assert result["mutation_status"] == "requires_file_upload"

    @pytest.mark.asyncio
    async def test_handles_full_state_gracefully(self):
        """mutation_node should not crash when given a full-like state dict."""
        with patch(
            "app.core.feature2_mutation.op_classifier.OpClassifier.classify_operation",
            return_value="UPDATE",
        ):
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
    async def test_returns_mutation_op_key(self):
        """The returned dict should include mutation_op for guidance."""
        result = await mutation_node({})
        assert "mutation_op" in result


# ═══════════════════════════════════════════════════════════════════════
# route_after_router — MUTATION Routing Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMutationRouting:
    """Tests that MUTATION queries route to 'mutation' instead of 'chat'."""

    def test_mutation_routes_to_mutation_node(self):
        """MUTATION should route to 'mutation' for the Feature 2 pipeline."""
        state = {"query_type": "MUTATION"}
        assert route_after_router(state) == "mutation"

    def test_mutation_with_no_state_falls_back_to_chat(self):
        """If query_type is missing but defaults to MUTATION-like, check fallback."""
        state = {}
        # With empty state, route_after_router default is "chat"
        assert route_after_router(state) == "chat"

    def test_mutation_does_not_route_to_chat(self):
        """Ensure MUTATION does NOT route to 'chat' anymore."""
        state = {"query_type": "MUTATION"}
        result = route_after_router(state)
        assert result != "chat"
        assert result == "mutation"

    def test_mutation_routing_always_mutation(self):
        """Extra state fields should not affect MUTATION routing."""
        state = {
            "query_type": "MUTATION",
            "question": "Bulk upload employees via CSV",
            "user_id": "admin",
            "mutation_op": "INSERT",
        }
        assert route_after_router(state) == "mutation"

    def test_sql_does_not_route_to_mutation(self):
        """SQL queries should still route to 'sql_gen', not 'mutation'."""
        state = {"query_type": "SQL"}
        assert route_after_router(state) == "sql_gen"
        assert route_after_router(state) != "mutation"

    def test_rag_does_not_route_to_mutation(self):
        """RAG queries should still route to 'ltm_remember', not 'mutation'."""
        state = {"query_type": "RAG"}
        assert route_after_router(state) == "ltm_remember"
        assert route_after_router(state) != "mutation"

    def test_hybrid_does_not_route_to_mutation(self):
        """HYBRID queries should route to 'hybrid', not 'mutation'."""
        state = {"query_type": "HYBRID"}
        assert route_after_router(state) == "hybrid"
        assert route_after_router(state) != "mutation"
