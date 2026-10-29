from typing import Any

import psycopg2
import psycopg2.extras

from app.config import get_settings
from app.core.audit_logger import AuditLogger
from app.core.feature1_sql.sql_validator import SQLValidator
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SQLExecutor:
    """
    Executes approved SQL queries and maintains structural transaction audit logs in PostgreSQL.
    """

    def __init__(self):
        settings = get_settings()
        self.conn_str = settings.supabase_db_url
        self.readonly_conn_str = settings.sql_readonly_db_url or self.conn_str
        self.audit = AuditLogger()

    @track(name="sql_executor_execute")
    def execute_and_log(
        self, query_id: str, question: str, sql: str
    ) -> list[dict[str, Any]]:
        """Run one approved read-only query and persist its audit record."""
        logger.info(f"Executing approved SQL query {query_id}")
        is_safe, error_message = SQLValidator().validate(sql)
        if not is_safe:
            raise ValueError(f"SQL safety validation failed: {error_message}")
        self._ensure_audit_table()

        conn = psycopg2.connect(self.readonly_conn_str)
        try:
            # PostgreSQL enforces read-only behavior independently of the
            # application-level validator.
            conn.set_session(readonly=True, autocommit=False)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                results = [dict(row) for row in cur.fetchall()]
            conn.commit()
        except Exception as exc:
            conn.rollback()
            conn.close()
            self._write_audit(query_id, question, sql, f"FAILED: {exc!s}")
            logger.error(f"SQL Execution failed: {exc}")
            raise ValueError(f"Failed to execute approved SQL: {exc!s}") from exc
        else:
            conn.close()

        self._write_audit(query_id, question, sql, "SUCCESS")
        logger.info(f"✓ SQL executed and logged successfully. Rows: {len(results)}")
        return results

    def _ensure_audit_table(self) -> None:
        conn = psycopg2.connect(self.conn_str)
        try:
            self.audit.ensure_table(conn)
        finally:
            conn.close()

    def _write_audit(self, query_id: str, question: str, sql: str, status: str) -> None:
        conn = psycopg2.connect(self.conn_str)
        try:
            self.audit.log(conn, query_id, question, sql, status)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
