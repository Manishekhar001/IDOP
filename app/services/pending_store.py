"""
Shared pending operations store — centralised singleton dicts.

Both the LangGraph graph nodes (nodes.py) and the API routes (sql.py, mutation.py)
need to read/write pending queries and mutations.  Using a shared module here
avoids the instance-isolation bug where the graph node creates its own local
TextToSQLService whose pending_queries dict is invisible to the route.

Each PendingStore persists its data to a Postgres table (via Supabase)
so pending approvals survive application restarts. An in-memory fallback
is used when the database is unavailable (local dev / tests).

Usage:
    from app.services.pending_store import pending_queries, pending_mutations
    pending_queries[query_id] = {"sql": "...", "status": "pending_approval", ...}
"""

import json
import logging
from typing import Any, Dict

from app.config import get_settings

logger = logging.getLogger(__name__)


class PendingStore(dict):
    """
    Dict subclass that persists entries to a Postgres table for crash resilience.

    Writes go to both in-memory dict (fast reads) and the database (survives restart).
    Reads check memory first, then fall back to the database.
    Deletes remove from both.
    """

    def __init__(self, table_name: str) -> None:
        super().__init__()
        self.table_name = table_name

    # ------------------------------------------------------------------
    # Database helpers (mirrors approval_gate pattern)
    # ------------------------------------------------------------------

    def _get_connection(self):
        """Get a Supabase connection. Returns None if unavailable."""
        settings = get_settings()
        db_url = settings.supabase_db_url or settings.database_url
        if not db_url:
            return None
        try:
            import psycopg2

            conn = psycopg2.connect(db_url, connect_timeout=2)
            return conn
        except Exception as e:
            logger.debug(f"DB connection failed (falling back to memory): {e}")
            return None

    def _ensure_table(self, conn) -> bool:
        """Create the pending store table if it does not exist."""
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR(100) PRIMARY KEY,
            payload JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Could not create table {self.table_name}: {e}")
            conn.rollback()
            return False

    # ------------------------------------------------------------------
    # Dict overrides with DB persistence
    # ------------------------------------------------------------------

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        # Always update memory for fast reads
        super().__setitem__(key, value)

        # Try to persist in database
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    payload = json.dumps(value, default=str)
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            INSERT INTO {self.table_name} (id, payload)
                            VALUES (%s, %s::jsonb)
                            ON CONFLICT (id)
                            DO UPDATE SET payload = EXCLUDED.payload
                            """,
                            (key, payload),
                        )
                    conn.commit()
                    logger.debug(
                        f"Persisted pending entry {key[:12]}... in {self.table_name}"
                    )
            except Exception as e:
                logger.error(f"Failed to persist pending entry in DB: {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()

    def __getitem__(self, key: str) -> Dict[str, Any]:
        try:
            return super().__getitem__(key)
        except KeyError:
            pass

        conn = self._get_connection()
        if not conn:
            raise KeyError(key)
        try:
            if self._ensure_table(conn):
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT payload FROM {self.table_name} WHERE id = %s",
                        (key,),
                    )
                    row = cur.fetchone()
                if row:
                    value = json.loads(row[0])
                    super().__setitem__(key, value)  # warm the cache
                    return value
        except Exception as e:
            logger.debug(f"DB read failed for pending entry: {e}")
        finally:
            if conn:
                conn.close()
        raise KeyError(key)

    def __delitem__(self, key: str) -> None:
        # Remove from memory
        try:
            super().__delitem__(key)
        except KeyError:
            pass

        # Remove from database
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM {self.table_name} WHERE id = %s",
                            (key,),
                        )
                    conn.commit()
                    logger.debug(
                        f"Removed pending entry {key[:12]}... from {self.table_name}"
                    )
            except Exception as e:
                logger.warning(f"Failed to remove pending entry from DB: {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()

    def __contains__(self, key: object) -> bool:
        # Check memory first, then DB
        if super().__contains__(key):
            return True
        try:
            self.__getitem__(str(key))
            return True
        except (KeyError, TypeError):
            return False

    def clear(self) -> None:
        # Clear memory
        super().clear()

        # Clear database
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(f"DELETE FROM {self.table_name}")
                    conn.commit()
                    logger.debug(f"Cleared all entries from {self.table_name}")
            except Exception as e:
                logger.warning(f"Failed to clear DB table {self.table_name}: {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()

    def _load_all_from_db(self) -> None:
        """Hydrate in-memory cache from DB (e.g., after restart)."""
        conn = self._get_connection()
        if not conn:
            return
        try:
            if self._ensure_table(conn):
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT id, payload FROM {self.table_name} ORDER BY created_at ASC"
                    )
                    rows = cur.fetchall()
                for row_id, payload_json in rows:
                    value = json.loads(payload_json)
                    super().__setitem__(row_id, value)
                if rows:
                    logger.info(
                        f"Recovered {len(rows)} pending entries from {self.table_name}"
                    )
        except Exception as e:
            logger.warning(f"Failed to load pending entries from DB: {e}")
        finally:
            if conn:
                conn.close()

    def items(self):
        if not super().__len__():
            self._load_all_from_db()
        return super().items()

    def __iter__(self):
        if not super().__len__():
            self._load_all_from_db()
        return super().__iter__()

    def __len__(self):
        if not super().__len__():
            self._load_all_from_db()
        return super().__len__()


# Shared pending SQL queries — used by graph nodes AND /sql routes
pending_queries: PendingStore = PendingStore(table_name="idop_pending_queries")

# Shared pending mutation sessions — used by mutation route
pending_mutations: PendingStore = PendingStore(table_name="idop_pending_mutations")


def reset_pending_store() -> None:
    """
    Clear both pending stores. Useful for test isolation to prevent
    cross-test contamination from shared module-level dicts.
    """
    pending_queries.clear()
    pending_mutations.clear()
