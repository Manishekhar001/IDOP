"""
Shared AuditLogger — consolidates the duplicate audit log logic from
``SQLExecutor`` (Feature 1) and ``MutationExecutor`` (Feature 2).

Both executors previously maintained identical ``_ensure_audit_table``
methods and repeated the same ``INSERT INTO idop_audit_logs`` pattern
in every success/failure path. This class centralises that logic.

Usage:

    from app.core.audit_logger import AuditLogger

    logger = AuditLogger()
    conn = psycopg2.connect(...)
    logger.ensure_table(conn)
    logger.log_success(conn, query_id, question, sql, "SUCCESS")
    logger.log_failure(conn, query_id, question, sql, error)
"""

import logging

logger = logging.getLogger("idop_app.audit_logger")


class AuditLogger:
    """Manages the ``idop_audit_logs`` table and provides helpers for
    writing success and failure audit entries."""

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS idop_audit_logs (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(100),
            question TEXT,
            sql_query TEXT,
            status VARCHAR(500),
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """

    INSERT_SQL = """
        INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
        VALUES (%s, %s, %s, %s)
    """

    def ensure_table(self, conn) -> None:
        """Create the ``idop_audit_logs`` table if it does not exist.

        Call this *before* disabling autocommit so the DDL commits on its
        own and doesn't break transaction atomicity.
        """
        try:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_TABLE_SQL)
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not create audit logs table: {e}")
            conn.rollback()

    def log(
        self,
        conn,
        query_id: str,
        question: str,
        sql_query: str,
        status: str,
    ) -> None:
        """Write a single audit log entry.

        The caller is responsible for committing or rolling back the
        transaction after calling this method.
        """
        with conn.cursor() as cur:
            cur.execute(self.INSERT_SQL, (query_id, question, sql_query, status))
