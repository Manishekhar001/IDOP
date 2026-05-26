import logging

logger = logging.getLogger("idop_app.sql_validator")


class SQLValidator:
    """
    Validates SQL queries for syntax safety and structural sanity.
    Prevents execution of forbidden operations.
    """

    # Forbidden SQL commands to prevent destructive operations in query pipeline
    FORBIDDEN_COMMANDS = {
        "DROP",
        "TRUNCATE",
        "ALTER",
        "GRANT",
        "REVOKE",
        "CREATE",
        "REPLACE",
    }

    def __init__(self):
        pass

    def validate(self, sql: str) -> tuple[bool, str]:
        """
        Validate the safety of the SQL string.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "SQL query is empty"

        cleaned_sql = sql.strip().upper()

        # Simple security parsing
        for command in self.FORBIDDEN_COMMANDS:
            # Check for command as separate word to avoid false positives
            if command in cleaned_sql.split():
                logger.warning(f"Validation failed: forbidden command '{command}' detected")
                return False, f"Destructive SQL command '{command}' is strictly forbidden."

        # Verify transaction safety
        if "COMMIT" in cleaned_sql.split() or "ROLLBACK" in cleaned_sql.split():
            return False, "Explicit transaction control (COMMIT/ROLLBACK) is not permitted inside user queries."

        return True, ""
