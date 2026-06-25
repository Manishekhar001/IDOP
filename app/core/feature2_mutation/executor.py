from typing import Any

import psycopg2

from app.config import get_settings
from app.core.audit_logger import AuditLogger
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MutationExecutor:
    """
    Executes approved database mutations inside a single rollback-safe database transaction.
    """

    def __init__(self):
        settings = get_settings()
        self.conn_str = settings.supabase_db_url
        self.audit = AuditLogger()

    @track(name="mutation_executor_insert")
    def execute_insert_transaction(
        self, mutation_id: str, table_name: str, sql: str, params: list[tuple[Any, ...]]
    ) -> int:
        """
        Execute bulk insert inside a single safe transaction.

        The audit table DDL is created BEFORE disabling autocommit so the
        CREATE TABLE IF NOT EXISTS cannot break transaction atomicity.
        """
        logger.info(
            f"Executing INSERT transaction for mutation {mutation_id} in '{table_name}'"
        )
        conn = psycopg2.connect(self.conn_str)
        self.audit.ensure_table(
            conn
        )  # DDL commits on its own before transaction starts
        conn.autocommit = False

        try:
            rows_inserted = 0
            with conn.cursor() as cur:
                # Execute in batch
                cur.executemany(sql, params)
                rows_inserted = cur.rowcount

            # Audit log
            self.audit.log(
                conn,
                mutation_id,
                f"Bulk INSERT to {table_name}",
                sql,
                f"SUCCESS: inserted {rows_inserted} rows",
            )
            conn.commit()
            conn.close()
            logger.info(
                f"✓ INSERT transaction committed successfully. {rows_inserted} rows inserted."
            )
            return rows_inserted

        except Exception as e:
            logger.error(f"INSERT Transaction failed - rolling back: {e}")
            conn.rollback()
            try:
                self.audit.log(
                    conn,
                    mutation_id,
                    f"Bulk INSERT to {table_name}",
                    sql,
                    f"FAILED: {e!s}",
                )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure log: {log_err}")
            conn.close()
            raise ValueError(f"All-or-Nothing bulk INSERT failed: {e!s}")

    @track(name="mutation_executor_update")
    def execute_updates_transaction(
        self,
        mutation_id: str,
        table_name: str,
        updates: list[tuple[str, tuple[Any, ...]]],
    ) -> int:
        """
        Execute updates inside a single safe transaction.
        """
        logger.info(
            f"Executing UPDATE transaction for mutation {mutation_id} in '{table_name}'"
        )
        conn = psycopg2.connect(self.conn_str)
        self.audit.ensure_table(conn)  # DDL before transaction start
        conn.autocommit = False

        try:
            rows_updated = 0
            with conn.cursor() as cur:
                for sql, params in updates:
                    cur.execute(sql, params)
                    rows_updated += cur.rowcount

            self.audit.log(
                conn,
                mutation_id,
                f"Bulk UPDATE to {table_name}",
                f"Updated {len(updates)} rows",
                f"SUCCESS: updated {rows_updated} rows",
            )
            conn.commit()
            conn.close()
            logger.info(
                f"✓ UPDATE transaction committed successfully. {rows_updated} rows updated."
            )
            return rows_updated

        except Exception as e:
            logger.error(f"UPDATE Transaction failed - rolling back: {e}")
            conn.rollback()
            try:
                self.audit.log(
                    conn,
                    mutation_id,
                    f"Bulk UPDATE to {table_name}",
                    "Update batch",
                    f"FAILED: {e!s}",
                )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure log: {log_err}")
            conn.close()
            raise ValueError(f"All-or-Nothing bulk UPDATE failed: {e!s}")

    @track(name="mutation_executor_delete")
    def execute_delete_transaction(
        self, mutation_id: str, table_name: str, sql: str, ids: list[Any]
    ) -> int:
        """
        Execute delete inside a single safe transaction.
        """
        logger.info(
            f"Executing DELETE transaction for mutation {mutation_id} in '{table_name}'"
        )
        conn = psycopg2.connect(self.conn_str)
        self.audit.ensure_table(conn)  # DDL before transaction start
        conn.autocommit = False

        try:
            rows_deleted = 0
            with conn.cursor() as cur:
                cur.execute(sql, tuple(ids))
                rows_deleted = cur.rowcount

            self.audit.log(
                conn,
                mutation_id,
                f"Bulk DELETE from {table_name}",
                sql,
                f"SUCCESS: deleted {rows_deleted} rows",
            )
            conn.commit()
            conn.close()
            logger.info(
                f"✓ DELETE transaction committed successfully. {rows_deleted} rows deleted."
            )
            return rows_deleted

        except Exception as e:
            logger.error(f"DELETE Transaction failed - rolling back: {e}")
            conn.rollback()
            try:
                self.audit.log(
                    conn,
                    mutation_id,
                    f"Bulk DELETE from {table_name}",
                    sql,
                    f"FAILED: {e!s}",
                )
                conn.commit()
            except Exception as log_err:
                logger.error(f"Failed to write failure log: {log_err}")
            conn.close()
            raise ValueError(f"All-or-Nothing bulk DELETE failed: {e!s}")
