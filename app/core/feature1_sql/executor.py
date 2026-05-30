import logging
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any
from app.opik import track
from app.config import get_settings

logger = logging.getLogger("idop_app.sql_executor")


class SQLExecutor:
    """
    Executes approved SQL queries and maintains structural transaction audit logs in PostgreSQL.
    """

    def __init__(self):
        settings = get_settings()
        self.conn_str = settings.supabase_db_url

    def _ensure_audit_table(self, conn) -> None:
        """Create audit log table if not exists"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS idop_audit_logs (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(100),
            question TEXT,
            sql_query TEXT,
            status VARCHAR(50),
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not create audit logs table: {e}")
            conn.rollback()

    @track(name="sql_executor_execute")
    def execute_and_log(self, query_id: str, question: str, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query and log to audit table.
        """
        logger.info(f"Executing approved SQL query {query_id}")
        conn = psycopg2.connect(self.conn_str)
        self._ensure_audit_table(conn)

        try:
            # Run Query
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                results = [dict(row) for row in cur.fetchall()]

            # Log to Audit
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (query_id, question, sql, "SUCCESS")
                )
            conn.commit()
            conn.close()
            logger.info(f"✓ SQL executed and logged successfully. Rows: {len(results)}")
            return results

        except Exception as e:
            logger.error(f"SQL Execution failed: {e}")
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (query_id, question, sql, f"FAILED: {str(e)}")
                    )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure audit log: {log_err}")
            conn.close()
            raise ValueError(f"Failed to execute approved SQL: {str(e)}")
