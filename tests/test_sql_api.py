"""
Unit tests for IDOP SQL API and schema updates.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.schemas import SQLResponse
from app.main import app


class TestSQLResponseSchema:
    """Tests the updated SQLResponse Pydantic schema."""

    def test_sql_response_accepts_token(self):
        """Test that the SQLResponse schema accepts and validates the token field."""
        response = SQLResponse(
            query_id="query_abc_123",
            question="How many users are there?",
            sql="SELECT COUNT(*) FROM users;",
            explanation="Counts users.",
            status="pending_approval",
            cache_hit=False,
            token="secret_token_123",
        )
        assert response.token == "secret_token_123"

    def test_sql_response_token_is_optional(self):
        """Test that the token field is optional and defaults to None."""
        response = SQLResponse(
            query_id="query_abc_123",
            question="How many users are there?",
            sql="SELECT COUNT(*) FROM users;",
            explanation="Counts users.",
            status="pending_approval",
            cache_hit=False,
        )
        assert response.token is None


class TestSQLApiEndpoints:
    """Tests the /sql/generate endpoint behavior with the crypto approval token."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("app.api.routes.sql.sql_service")
    @patch("app.api.routes.sql.gate")
    @patch("app.api.routes.sql.shared_pending_queries", new_callable=dict)
    def test_generate_sql_returns_token_and_updates_cache(
        self, mock_shared_pending, mock_gate, mock_sql_service, client
    ):
        """Test that /sql/generate creates a session, returns the token, and updates pending_queries."""
        # Override auth to return a test user
        app.dependency_overrides[get_current_user] = lambda: {"sub": "test@example.com", "role": "user"}

        try:
            # Setup mocks
            mock_query_id = "test-query-id-123"
            mock_sql_service.generate_sql_for_approval = AsyncMock(
                return_value={
                    "query_id": mock_query_id,
                    "question": "Show all customers",
                    "sql": "SELECT * FROM customers;",
                    "explanation": "Selects all customers",
                    "status": "pending_approval",
                    "cache_hit": False,
                }
            )

            # Pre-populate shared pending with the query_id (mimicking what sql_service would have returned)
            mock_shared_pending[mock_query_id] = {
                "question": "Show all customers",
                "sql": "SELECT * FROM customers;",
                "status": "pending_approval",
                "cache_hit": False,
            }

            mock_token = "mock-crypto-approval-token-999"
            mock_gate.generate_session.return_value = mock_token

            # Call the endpoint with the new Pydantic body
            response = client.post(
                "/sql/generate",
                json={"question": "Show all customers", "vanna_temperature": 0.0},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify the returned values
            assert data["query_id"] == mock_query_id
            assert data["token"] == mock_token

            # Verify gate was called with query_id
            mock_gate.generate_session.assert_called_once_with(mock_query_id)

            # Verify the shared pending queries entry was updated with the token
            assert mock_shared_pending[mock_query_id]["token"] == mock_token
        finally:
            app.dependency_overrides.pop(get_current_user, None)
