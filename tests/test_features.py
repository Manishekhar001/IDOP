"""
Unit tests for IDOP Feature pipelines.

- Feature 1 (SQL): SQLValidator safety checks, ApprovalGate token management.
- Feature 2 (Mutations): RuleValidator business rules enforcement.
"""

import json

import pytest

from app.core.approval_gate import ApprovalGate
from app.core.feature1_sql.sql_validator import SQLValidator
from app.core.feature2_mutation.rule_validator import RuleValidator

# ═══════════════════════════════════════════════════════════════════════
# Feature 1: SQL Validator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSQLValidator:
    """Tests for the SQL safety validator preventing destructive operations."""

    @pytest.fixture
    def validator(self):
        return SQLValidator()

    # ----- Safe Queries -----

    def test_valid_select_query(self, validator):
        """Test that a simple SELECT query passes validation."""
        is_valid, error = validator.validate("SELECT * FROM products WHERE price > 100")
        assert is_valid is True
        assert error == ""

    def test_valid_select_with_join(self, validator):
        """Test that a SELECT query with JOINs passes validation."""
        sql = """
        SELECT p.name, o.quantity
        FROM products p
        JOIN orders o ON p.id = o.product_id
        WHERE o.status = 'Delivered'
        """
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_valid_select_with_aggregation(self, validator):
        """Test that a SELECT with GROUP BY/HAVING passes validation."""
        sql = "SELECT segment, COUNT(*) FROM customers GROUP BY segment HAVING COUNT(*) > 5"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_valid_select_with_subquery(self, validator):
        """Test that a SELECT with a subquery passes validation."""
        sql = "SELECT * FROM products WHERE id IN (SELECT product_id FROM orders)"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    # ----- Forbidden Commands -----

    def test_drop_table_blocked(self, validator):
        """Test that DROP TABLE is blocked."""
        is_valid, error = validator.validate("DROP TABLE products")
        assert is_valid is False
        assert "DROP" in error

    def test_truncate_blocked(self, validator):
        """Test that TRUNCATE is blocked."""
        is_valid, error = validator.validate("TRUNCATE TABLE orders")
        assert is_valid is False
        assert "TRUNCATE" in error

    def test_alter_table_blocked(self, validator):
        """Test that ALTER TABLE is blocked."""
        is_valid, error = validator.validate(
            "ALTER TABLE customers ADD COLUMN phone VARCHAR(20)"
        )
        assert is_valid is False
        assert "ALTER" in error

    def test_grant_blocked(self, validator):
        """Test that GRANT is blocked."""
        is_valid, error = validator.validate("GRANT ALL ON products TO public")
        assert is_valid is False
        assert "GRANT" in error

    def test_revoke_blocked(self, validator):
        """Test that REVOKE is blocked."""
        is_valid, error = validator.validate("REVOKE SELECT ON products FROM public")
        assert is_valid is False
        assert "REVOKE" in error

    def test_create_table_blocked(self, validator):
        """Test that CREATE TABLE is blocked."""
        is_valid, error = validator.validate("CREATE TABLE malicious (id INT)")
        assert is_valid is False
        assert "CREATE" in error

    def test_replace_blocked(self, validator):
        """Test that REPLACE is blocked."""
        is_valid, error = validator.validate(
            "REPLACE INTO products VALUES (1, 'test', 99.99)"
        )
        assert is_valid is False
        assert "REPLACE" in error

    # ----- Transaction Control -----

    def test_commit_blocked(self, validator):
        """Test that explicit COMMIT is blocked."""
        is_valid, error = validator.validate("COMMIT")
        assert is_valid is False
        assert "COMMIT" in error or "transaction" in error.lower()

    def test_rollback_blocked(self, validator):
        """Test that explicit ROLLBACK is blocked."""
        is_valid, error = validator.validate("ROLLBACK")
        assert is_valid is False
        assert "ROLLBACK" in error or "transaction" in error.lower()

    # ----- Edge Cases -----

    def test_empty_query_rejected(self, validator):
        """Test that an empty SQL query is rejected."""
        is_valid, error = validator.validate("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_none_query_rejected(self, validator):
        """Test that a None SQL query is rejected."""
        is_valid, error = validator.validate(None)
        assert is_valid is False

    def test_whitespace_only_rejected(self, validator):
        """Test that whitespace-only SQL is rejected."""
        is_valid, error = validator.validate("   ")
        assert is_valid is False

    def test_insert_is_blocked(self, validator):
        """Test that INSERT is blocked in the read-only query pipeline."""
        is_valid, error = validator.validate(
            "INSERT INTO products VALUES (1, 'Widget', 9.99)"
        )
        assert is_valid is False
        assert "SELECT" in error or "INSERT" in error

    def test_update_is_blocked(self, validator):
        """Test that UPDATE is blocked in the read-only query pipeline."""
        is_valid, error = validator.validate(
            "UPDATE products SET price = 10.99 WHERE id = 1"
        )
        assert is_valid is False
        assert "SELECT" in error or "UPDATE" in error

    def test_delete_is_blocked(self, validator):
        """Test that DELETE is blocked in the read-only query pipeline."""
        is_valid, error = validator.validate(
            "DELETE FROM orders WHERE status = 'Cancelled'"
        )
        assert is_valid is False
        assert "SELECT" in error or "DELETE" in error


# ═══════════════════════════════════════════════════════════════════════
# Feature 1: Approval Gate Tests
# ═══════════════════════════════════════════════════════════════════════


class TestApprovalGate:
    """Tests for the cryptographic session-based approval gate."""

    @pytest.fixture
    def gate(self):
        return ApprovalGate(
            table_name="test_tokens",
            session_column="query_id",
            logger_name="test_gate",
        )

    def test_generate_session_returns_token(self, gate):
        """Test that generating a session returns a hex token."""
        token = gate.generate_session("query_001")
        assert isinstance(token, str)
        assert len(token) == 64  # secrets.token_hex(32) → 64 hex chars

    def test_generate_session_stores_in_active_sessions(self, gate):
        """Test that the generated token is stored in active_sessions."""
        token = gate.generate_session("query_002")
        assert "query_002" in gate.active_sessions
        assert gate.active_sessions["query_002"] == token

    def test_verify_correct_token_succeeds(self, gate):
        """Test that verification with the correct token succeeds."""
        token = gate.generate_session("query_003")
        result = gate.verify_and_close_session("query_003", token)
        assert result is True

    def test_verify_removes_session_after_success(self, gate):
        """Test that the session is removed after successful verification (single-use)."""
        token = gate.generate_session("query_004")
        gate.verify_and_close_session("query_004", token)
        assert "query_004" not in gate.active_sessions

    def test_verify_wrong_token_fails(self, gate):
        """Test that verification with an incorrect token fails."""
        gate.generate_session("query_005")
        result = gate.verify_and_close_session("query_005", "wrong_token_value")
        assert result is False

    def test_verify_wrong_token_does_not_remove_session(self, gate):
        """Test that a failed verification does NOT remove the session."""
        gate.generate_session("query_006")
        gate.verify_and_close_session("query_006", "bad_token")
        assert "query_006" in gate.active_sessions

    def test_verify_nonexistent_query_id_fails(self, gate):
        """Test that verifying a non-existent query ID fails."""
        result = gate.verify_and_close_session("no_such_query", "some_token")
        assert result is False

    def test_double_verification_fails(self, gate):
        """Test that using the same token twice fails (single-use enforcement)."""
        token = gate.generate_session("query_007")
        assert gate.verify_and_close_session("query_007", token) is True
        assert gate.verify_and_close_session("query_007", token) is False

    def test_multiple_independent_sessions(self, gate):
        """Test that multiple concurrent sessions are independent."""
        token_a = gate.generate_session("query_a")
        token_b = gate.generate_session("query_b")

        assert token_a != token_b
        assert gate.verify_and_close_session("query_a", token_a) is True
        assert gate.verify_and_close_session("query_b", token_b) is True

    def test_token_uniqueness(self, gate):
        """Test that each generated token is cryptographically unique."""
        tokens = set()
        for i in range(100):
            token = gate.generate_session(f"unique_query_{i}")
            tokens.add(token)
        assert len(tokens) == 100


# ═══════════════════════════════════════════════════════════════════════
# Feature 2: Rule Validator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRuleValidator:
    """Tests for the business rules validator using rules.json configuration."""

    @pytest.fixture
    def rules_file(self, tmp_path):
        """Create a temporary rules.json for testing."""
        rules = {
            "max_bulk_rows": 5,
            "allowed_mutation_tables": ["products", "customers", "orders"],
            "require_confirmation_ops": ["DELETE"],
            "field_validation_rules": {
                "products": {
                    "price": {
                        "type": "numeric",
                        "min": 0.01,
                        "message": "Product price must be greater than zero.",
                    },
                    "stock_quantity": {
                        "type": "integer",
                        "min": 0,
                        "message": "Stock quantity cannot be negative.",
                    },
                },
                "customers": {
                    "email": {
                        "type": "regex",
                        "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$",
                        "message": "Customer email must be a valid email format.",
                    },
                    "segment": {
                        "type": "enum",
                        "allowed": ["SMB", "Enterprise", "Individual"],
                        "message": "Segment must be one of: SMB, Enterprise, Individual.",
                    },
                },
                "orders": {
                    "total_amount": {
                        "type": "numeric",
                        "min": 0.0,
                        "message": "Order total amount cannot be negative.",
                    },
                    "status": {
                        "type": "enum",
                        "allowed": ["Pending", "Delivered", "Cancelled", "Processing"],
                        "message": "Status must be one of: Pending, Delivered, Cancelled, Processing.",
                    },
                },
            },
        }
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps(rules, indent=2))
        return str(rules_path)

    @pytest.fixture
    def validator(self, rules_file):
        return RuleValidator(rules_path=rules_file)

    # ----- Valid Rows -----

    def test_valid_product_rows(self, validator):
        """Test that valid product rows pass validation."""
        rows = [
            {"price": 29.99, "stock_quantity": 100},
            {"price": 0.01, "stock_quantity": 0},
        ]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is True
        assert errors == []

    def test_valid_customer_rows(self, validator):
        """Test that valid customer rows pass validation."""
        rows = [
            {"email": "test@example.com", "segment": "SMB"},
            {"email": "user@company.org", "segment": "Enterprise"},
        ]
        is_valid, errors = validator.validate_rows("customers", rows)
        assert is_valid is True
        assert errors == []

    def test_valid_order_rows(self, validator):
        """Test that valid order rows pass validation."""
        rows = [
            {"total_amount": 150.50, "status": "Pending"},
            {"total_amount": 0.0, "status": "Delivered"},
        ]
        is_valid, errors = validator.validate_rows("orders", rows)
        assert is_valid is True

    # ----- Invalid Rows -----

    def test_negative_product_price_rejected(self, validator):
        """Test that a negative product price is rejected."""
        rows = [{"price": -5.00, "stock_quantity": 10}]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is False
        assert len(errors) == 1
        assert "price" in errors[0].lower() or "greater than zero" in errors[0].lower()

    def test_negative_stock_quantity_rejected(self, validator):
        """Test that a negative stock quantity is rejected."""
        rows = [{"price": 10.00, "stock_quantity": -1}]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is False

    def test_invalid_email_rejected(self, validator):
        """Test that an invalid email format is rejected."""
        rows = [{"email": "not-an-email", "segment": "SMB"}]
        is_valid, errors = validator.validate_rows("customers", rows)
        assert is_valid is False
        assert "email" in errors[0].lower()

    def test_invalid_segment_rejected(self, validator):
        """Test that an invalid customer segment is rejected."""
        rows = [{"email": "test@example.com", "segment": "VIP"}]
        is_valid, errors = validator.validate_rows("customers", rows)
        assert is_valid is False
        assert "Segment" in errors[0] or "segment" in errors[0].lower()

    def test_invalid_order_status_rejected(self, validator):
        """Test that an invalid order status is rejected."""
        rows = [{"total_amount": 50.00, "status": "Shipped"}]
        is_valid, errors = validator.validate_rows("orders", rows)
        assert is_valid is False
        assert "Status" in errors[0] or "status" in errors[0].lower()

    def test_negative_order_amount_rejected(self, validator):
        """Test that a negative order amount is rejected."""
        rows = [{"total_amount": -10.00, "status": "Pending"}]
        is_valid, errors = validator.validate_rows("orders", rows)
        assert is_valid is False

    # ----- Bulk Row Limit -----

    def test_exceeding_max_bulk_rows_rejected(self, validator):
        """Test that payloads exceeding max_bulk_rows are rejected."""
        rows = [{"price": 10.0, "stock_quantity": 1} for _ in range(10)]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is False
        assert "exceeds" in errors[0].lower() or "maximum" in errors[0].lower()

    def test_at_max_bulk_rows_accepted(self, validator):
        """Test that payloads at exactly max_bulk_rows are accepted."""
        rows = [{"price": 10.0, "stock_quantity": 1} for _ in range(5)]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is True

    # ----- Unknown Tables -----

    def test_unknown_table_passes_validation(self, validator):
        """Test that rows for an unknown table pass (no rules to enforce)."""
        rows = [{"any_field": "any_value"}]
        is_valid, errors = validator.validate_rows("unknown_table", rows)
        assert is_valid is True
        assert errors == []

    # ----- Null Handling -----

    def test_null_fields_are_skipped(self, validator):
        """Test that null/missing fields are skipped without error."""
        rows = [{"price": None, "stock_quantity": 10}]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is True

    def test_missing_optional_fields(self, validator):
        """Test that rows with missing fields (not present at all) are valid."""
        rows = [{"stock_quantity": 5}]  # 'price' key not present
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is True

    # ----- Non-Numeric Type Errors -----

    def test_non_numeric_price_rejected(self, validator):
        """Test that a non-numeric price value is caught as a type error."""
        rows = [{"price": "not_a_number", "stock_quantity": 10}]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is False
        assert "numeric" in errors[0].lower()

    def test_non_integer_stock_quantity_rejected(self, validator):
        """Test that a non-integer stock quantity is caught as a type error."""
        rows = [{"price": 10.0, "stock_quantity": "abc"}]
        is_valid, errors = validator.validate_rows("products", rows)
        assert is_valid is False
        assert "integer" in errors[0].lower()


# ═══════════════════════════════════════════════════════════════════════
# Extension Sync Validation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestExtensionSync:
    """
    Validates that FileValidator.ALLOWED_EXTENSIONS keys stay in sync
    with DocumentProcessor.SUPPORTED_EXTENSIONS.

    DocumentProcessor.SUPPORTED_EXTENSIONS is the authoritative allow-list
    for file types. FileValidator.ALLOWED_EXTENSIONS must mirror it exactly
    so that validation rejects the same extensions that the processor cannot
    handle.
    """

    def test_validator_extensions_match_document_processor(self):
        """
        FileValidator.ALLOWED_EXTENSIONS keys must be an exact match
        with DocumentProcessor.SUPPORTED_EXTENSIONS.
        """
        # Lazy import: DocumentProcessor triggers langchain imports (slow).
        from app.core.document_processor import DocumentProcessor
        from app.utils.validators import FileValidator

        validator_keys = set(FileValidator.ALLOWED_EXTENSIONS.keys())
        processor_exts = DocumentProcessor.SUPPORTED_EXTENSIONS

        missing_in_validator = processor_exts - validator_keys
        extra_in_validator = validator_keys - processor_exts

        assert validator_keys == processor_exts, (
            f"Extension sets out of sync!\n"
            f"  Missing from FileValidator: {missing_in_validator}\n"
            f"  Extra in FileValidator: {extra_in_validator}\n"
            f"  FileValidator keys: {sorted(validator_keys)}\n"
            f"  DocumentProcessor: {sorted(processor_exts)}"
        )
