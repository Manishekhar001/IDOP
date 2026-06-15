import logging
import re
from typing import ClassVar

from app.opik import track

logger = logging.getLogger("idop_app.sql_validator")


class SQLValidator:
    """
    Validates SQL queries for syntax safety and structural sanity.
    Prevents execution of forbidden operations.
    """

    # Forbidden SQL commands to prevent destructive/write operations in query pipeline
    FORBIDDEN_COMMANDS: ClassVar[set[str]] = {
        "DROP",
        "TRUNCATE",
        "ALTER",
        "GRANT",
        "REVOKE",
        "CREATE",
        "REPLACE",
        "DELETE",
        "UPDATE",
        "INSERT",
        "EXECUTE",
        "EXEC",
    }

    @track(name="sql_validator_validate")
    def validate(self, sql: str) -> tuple[bool, str]:
        """
        Validate the safety and read-only nature of the SQL string.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "SQL query is empty"

        cleaned_sql = sql.strip().upper()

        # Simple security parsing — use word-boundary regex to avoid false positives
        for command in self.FORBIDDEN_COMMANDS:
            if re.search(r"\b" + re.escape(command) + r"\b", cleaned_sql):
                logger.warning(
                    f"Validation failed: forbidden command '{command}' detected"
                )
                return (
                    False,
                    f"Destructive or mutating SQL command '{command}' is strictly forbidden.",
                )

        # Verify transaction safety — use word-boundary regex
        if re.search(r"\bCOMMIT\b", cleaned_sql) or re.search(
            r"\bROLLBACK\b", cleaned_sql
        ):
            return (
                False,
                "Explicit transaction control (COMMIT/ROLLBACK) is not permitted inside user queries.",
            )

        # Enforce that all user SQL queries must start with SELECT
        if not cleaned_sql.startswith("SELECT"):
            logger.warning(
                "Validation failed: query does not begin with SELECT (read-only enforcement)"
            )
            return (
                False,
                "Only read-only SELECT queries are permitted in Feature 1 NL-to-SQL.",
            )

        return True, ""
