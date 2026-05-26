"""
Unit tests for IDOP enhanced API responses and detail schemas.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.api.schemas import ChatResponse, MutationResponse
from app.main import app


class TestEnhancedSchemas:
    """Tests the detailed schema definitions for ChatResponse and MutationResponse."""

    def test_chat_response_accepts_detailed_fields(self):
        """Test that ChatResponse model accepts and validates new rich operational fields."""
        response = ChatResponse(
            question="Compare sales to strategy guidelines",
            answer="Here is the report...",
            processing_time_ms=120.5,
            query_type="HYBRID",
            ltm_context="User is based in Canada.",
            mutation_id="mut-123",
            mutation_table="products",
            mutation_op="INSERT",
            mutation_status="pending_approval",
            mutation_result_count=5,
            approval_token="secure_token_999"
        )
        assert response.query_type == "HYBRID"
        assert response.ltm_context == "User is based in Canada."
        assert response.mutation_id == "mut-123"
        assert response.mutation_table == "products"
        assert response.mutation_op == "INSERT"
        assert response.mutation_status == "pending_approval"
        assert response.mutation_result_count == 5
        assert response.approval_token == "secure_token_999"

    def test_mutation_response_accepts_token(self):
        """Test that MutationResponse model accepts and validates the cryptographic token field."""
        response = MutationResponse(
            mutation_id="mut-abc",
            table_name="customers",
            op_type="UPDATE",
            row_count=10,
            status="pending_approval",
            mappings={"Name": "name"},
            errors=[],
            token="mutation_secret_token_123"
        )
        assert response.token == "mutation_secret_token_123"


class TestEnhancedApiEndpoints:
    """Tests that routes return detailed payloads and serialize new fields properly."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_chat_endpoint_returns_rich_fields(self, client):
        """Test that /chat returns the 5-path router classifications, memory context, and approval tokens."""
        mock_engine = MagicMock()
        
        from app.api.routes.chat import get_engine
        app.dependency_overrides[get_engine] = lambda: mock_engine
        
        try:
            # Setup mock return value with all rich fields populated
            mock_engine.aquery = AsyncMock(return_value={
                "answer": "This is a detailed response.",
                "sources": [],
                "crag_verdict": "CORRECT",
                "issup": "fully_supported",
                "evidence": [],
                "retries": 0,
                "rewrite_tries": 0,
                "query_type": "HYBRID",
                "ltm_context": "User prefers CSV reports.",
                "sql_query": "SELECT * FROM orders;",
                "sql_status": "executed",
                "approval_token": "token-12345",
                "mutation_id": "mut-777",
                "mutation_table": "orders",
                "mutation_op": "UPDATE",
                "mutation_status": "pending_approval",
                "mutation_result_count": 0
            })

            payload = {
                "question": "Execute hybrid query",
                "thread_id": "thread-111",
                "user_id": "user-222",
                "include_sources": False
            }
            
            response = client.post("/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Assert detailed fields are serialized in the response JSON
            assert data["query_type"] == "HYBRID"
            assert data["ltm_context"] == "User prefers CSV reports."
            assert data["approval_token"] == "token-12345"
            assert data["mutation_id"] == "mut-777"
            assert data["mutation_table"] == "orders"
            assert data["mutation_op"] == "UPDATE"
            assert data["mutation_status"] == "pending_approval"
        finally:
            app.dependency_overrides.pop(get_engine, None)
