import logging
import psycopg2
from typing import List, Dict, Any, Tuple
from app.config import get_settings

logger = logging.getLogger("idop_app.mutation_executor")


class MutationExecutor:
    """
    Executes approved database mutations inside a single rollback-safe database transaction.
    """

    def __init__(self):
        settings = get_settings()
        self.conn_str = settings.supabase_db_url

    def _ensure_audit_table(self, conn) -> None:
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

    def execute_insert_transaction(self, mutation_id: str, table_name: str, sql: str, params: List[Tuple[Any, ...]]) -> int:
        """
        Execute bulk insert inside a single safe transaction.
        """
        logger.info(f"Executing INSERT transaction for mutation {mutation_id} in '{table_name}'")
        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = False  # Ensure transaction block
        self._ensure_audit_table(conn)

        try:
            rows_inserted = 0
            with conn.cursor() as cur:
                # Execute in batch
                cur.executemany(sql, params)
                rows_inserted = cur.rowcount

            # Audit log
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (mutation_id, f"Bulk INSERT to {table_name}", sql, f"SUCCESS: inserted {rows_inserted} rows")
                )
            conn.commit()
            conn.close()
            logger.info(f"✓ INSERT transaction committed successfully. {rows_inserted} rows inserted.")
            return rows_inserted

        except Exception as e:
            logger.error(f"INSERT Transaction failed - rolling back: {e}")
            conn.rollback()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (mutation_id, f"Bulk INSERT to {table_name}", sql, f"FAILED: {str(e)}")
                    )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure log: {log_err}")
            conn.close()
            raise ValueError(f"All-or-Nothing bulk INSERT failed: {str(e)}")

    def execute_updates_transaction(self, mutation_id: str, table_name: str, updates: List[Tuple[str, Tuple[Any, ...]]]) -> int:
        """
        Execute updates inside a single safe transaction.
        """
        logger.info(f"Executing UPDATE transaction for mutation {mutation_id} in '{table_name}'")
        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = False
        self._ensure_audit_table(conn)

        try:
            rows_updated = 0
            with conn.cursor() as cur:
                for sql, params in updates:
                    cur.execute(sql, params)
                    rows_updated += cur.rowcount

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (mutation_id, f"Bulk UPDATE to {table_name}", f"Updated {len(updates)} rows", f"SUCCESS: updated {rows_updated} rows")
                )
            conn.commit()
            conn.close()
            logger.info(f"✓ UPDATE transaction committed successfully. {rows_updated} rows updated.")
            return rows_updated

        except Exception as e:
            logger.error(f"UPDATE Transaction failed - rolling back: {e}")
            conn.rollback()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (mutation_id, f"Bulk UPDATE to {table_name}", "Update batch", f"FAILED: {str(e)}")
                    )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure log: {log_err}")
            conn.close()
            raise ValueError(f"All-or-Nothing bulk UPDATE failed: {str(e)}")

    def execute_delete_transaction(self, mutation_id: str, table_name: str, sql: str, ids: List[Any]) -> int:
        """
        Execute delete inside a single safe transaction.
        """
        logger.info(f"Executing DELETE transaction for mutation {mutation_id} in '{table_name}'")
        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = False
        self._ensure_audit_table(conn)

        try:
            rows_deleted = 0
            with conn.cursor() as cur:
                cur.execute(sql, tuple(ids))
                rows_deleted = cur.rowcount

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (mutation_id, f"Bulk DELETE from {table_name}", sql, f"SUCCESS: deleted {rows_deleted} rows")
                )
            conn.commit()
            conn.close()
            logger.info(f"✓ DELETE transaction committed successfully. {rows_deleted} rows deleted.")
            return rows_deleted

        except Exception as e:
            logger.error(f"DELETE Transaction failed - rolling back: {e}")
            conn.rollback()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO idop_audit_logs (query_id, question, sql_query, status)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (mutation_id, f"Bulk DELETE from {table_name}", sql, f"FAILED: {str(e)}")
                    )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure log: {log_err}")
            conn.close()
            raise ValueError(f"All-or-Nothing bulk DELETE failed: {str(e)}")
