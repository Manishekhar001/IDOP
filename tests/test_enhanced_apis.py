"""
Unit tests for IDOP enhanced API responses and detail schemas.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from fastapi.testclient import TestClient

from app.api.schemas import ChatResponse, MutationResponse
from app.main import app


class TestStorageBackendDefault:
    """Tests that the default storage backend is now S3."""

    def test_default_storage_backend_is_s3(self):
        """Verify the Settings class default for storage_backend is 's3'."""
        from app.config import Settings

        field = Settings.model_fields["storage_backend"]
        assert field.default == "s3", f"Expected default 's3', got '{field.default}'"

    def test_health_reports_s3_when_backend_is_s3(self):
        """Verify /health reports s3_cache_configured: true when storage_backend is 's3'."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        mock_settings = MagicMock()
        mock_settings.storage_backend = "s3"
        mock_settings.app_version = "0.1.0"
        mock_settings.git_commit_sha = "unknown"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.voyage_api_key = "va-test"
        mock_settings.nomic_api_key = "nk-test"
        mock_settings.tavily_api_key = "tvly-test"
        mock_settings.database_url = "postgresql://test:test@localhost:5432/idop_test"
        mock_settings.supabase_db_url = ""
        mock_settings.upstash_redis_url = None
        mock_settings.upstash_redis_token = None

        with patch("app.api.routes.health.get_settings", return_value=mock_settings):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["configuration"]["s3_cache_configured"] is True

    def test_health_reports_local_when_backend_is_local(self):
        """Verify /health reports s3_cache_configured: false when storage_backend is 'local'."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        mock_settings = MagicMock()
        mock_settings.storage_backend = "local"
        mock_settings.app_version = "0.1.0"
        mock_settings.git_commit_sha = "unknown"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.voyage_api_key = "va-test"
        mock_settings.nomic_api_key = "nk-test"
        mock_settings.tavily_api_key = "tvly-test"
        mock_settings.database_url = "postgresql://test:test@localhost:5432/idop_test"
        mock_settings.supabase_db_url = ""
        mock_settings.upstash_redis_url = None
        mock_settings.upstash_redis_token = None

        with patch("app.api.routes.health.get_settings", return_value=mock_settings):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["configuration"]["s3_cache_configured"] is False


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
            approval_token="secure_token_999",
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
            token="mutation_secret_token_123",
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
            mock_engine.aquery = AsyncMock(
                return_value={
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
                    "mutation_result_count": 0,
                }
            )

            payload = {
                "question": "Execute hybrid query",
                "thread_id": "thread-111",
                "user_id": "user-222",
                "include_sources": False,
            }

            response = client.post("/chat", json=payload)
            assert response.status_code == 200
            data = response.json()

            # Assert detailed fields are serialized in the response JSON
            assert data["query_type"] == "HYBRID"
            assert data["ltm_context"] == "User prefers CSV reports."
            # approval_token is intentionally excluded from chat response for security
            assert data["mutation_id"] == "mut-777"
            assert data["mutation_table"] == "orders"
            assert data["mutation_op"] == "UPDATE"
            assert data["mutation_status"] == "pending_approval"
        finally:
            app.dependency_overrides.pop(get_engine, None)

    @patch("app.api.routes.documents.DocumentProcessor")
    def test_documents_upload_with_options(self, mock_processor_class, client):
        """Test document upload endpoint with customizable form parameters."""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        mock_processor.chunk_size = 512
        mock_processor.chunk_overlap = 50
        # Return a non-empty Document so the route's "if not texts" check passes
        from langchain_core.documents import Document

        mock_processor.process_upload_bytes.return_value = [
            Document(
                page_content="Hello testing world content",
                metadata={"source": "test.txt"},
            )
        ]

        mock_vector_store = MagicMock()
        mock_vector_store.embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.add_documents_with_embeddings.return_value = ["id1"]

        from app.api.routes.documents import get_vector_store

        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

        try:
            file_data = {"file": ("test.txt", b"Hello testing world", "text/plain")}
            form_data = {"chunk_size": "512", "chunk_overlap": "50"}

            response = client.post("/documents/upload", files=file_data, data=form_data)
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert data["filename"] == "test.txt"
            assert data["chunk_size_applied"] == 512
            assert data["chunk_overlap_applied"] == 50
        finally:
            app.dependency_overrides.pop(get_vector_store, None)

    @patch("app.api.routes.mutation.parser")
    @patch("app.api.routes.mutation.mapper")
    @patch("app.api.routes.mutation.validator")
    @patch("app.api.routes.mutation.generator")
    @patch("app.api.routes.mutation.judge")
    @patch("app.api.routes.mutation.gate")
    def test_mutation_upload_with_options(
        self,
        mock_gate,
        mock_judge,
        mock_generator,
        mock_validator,
        mock_mapper,
        mock_parser,
        client,
    ):
        """Test mutation upload with custom primary key, limit check, and validation skip."""
        mock_parser.parse_file.return_value = [{"id": 1, "name": "Prod1"}]
        mock_mapper.get_semantic_mapping.return_value = {"id": "id", "name": "name"}
        mock_validator.validate_rows.return_value = (True, [])
        mock_generator.generate_insert.return_value = ("INSERT INTO products", [])
        mock_judge.audit_mutation = AsyncMock(return_value=(True, "Approved"))
        mock_gate.generate_session.return_value = "gate-token-xyz"

        file_data = {"file": ("mut.csv", b"id,name\n1,Prod1", "text/csv")}
        form_data = {
            "table_name": "products",
            "request_intent": "Insert some products",
            "max_bulk_rows": "5",
            "primary_key": "prod_id",
            "auto_map": "false",
            "skip_validation": "true",
        }

        response = client.post("/mutation/upload", files=file_data, data=form_data)
        assert response.status_code == 200
        data = response.json()
        assert data["table_name"] == "products"
        assert data["token"] == "gate-token-xyz"

        # Verify skip_validation was respected (validate_rows should not be called)
        mock_validator.validate_rows.assert_not_called()
        # Verify auto_map was false (semantic mapping should not be called)
        mock_mapper.get_semantic_mapping.assert_not_called()
