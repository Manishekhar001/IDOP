"""
Shared pending operations store — Redis-backed singleton dicts for cross-worker consistency.

Both the LangGraph graph nodes (nodes.py) and the API routes (sql.py, mutation.py)
need to read/write pending queries and mutations.  Using a shared module here
avoids the instance-isolation bug where the graph node creates its own local
TextToSQLService whose pending_queries dict is invisible to the route.

Redis is the canonical store so that all uvicorn workers see the same pending
data.  If Redis is unavailable, we fall back to the in-memory dict + Postgres
persistence (matching the original behaviour before the Redis migration).

Each pending entry auto-expires after 1 hour (TTL) to prevent stale entries.

Usage:
    from app.services.pending_store import pending_queries, pending_mutations
    pending_queries[query_id] = {"sql": "...", "status": "pending_approval", ...}
    info = pending_queries[query_id]
"""

import json
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────
_PENDING_TTL = 3600  # 1 hour — pending items auto-expire


class PendingStore(dict):
    """
    Dict subclass that uses **Redis** as its canonical store so all uvicorn
    workers share the same data.

    Layered persistence strategy:
      1. **Redis** (fast, shared across workers, auto-expiry via TTL)
      2. **In-memory dict** + Postgres table (fallback when Redis is down)

    The dict interface is preserved so all existing callers continue to work
    without changes.
    """

    def __init__(self, table_name: str) -> None:
        super().__init__()
        self.table_name = table_name
        self._redis: Any = None
        self._redis_available = False
        self._init_redis()

    # ------------------------------------------------------------------
    # Redis initialisation
    # ------------------------------------------------------------------

    def _init_redis(self) -> None:
        """Try to connect to Upstash Redis using the project's existing config."""
        settings = get_settings()
        redis_url = settings.upstash_redis_url
        redis_token = settings.upstash_redis_token
        if not redis_url or not redis_token:
            logger.info(
                "PendingStore: Upstash Redis not configured — "
                "falling back to in-memory + Postgres"
            )
            return
        try:
            from upstash_redis import Redis

            self._redis = Redis(url=redis_url, token=redis_token)
            self._redis.ping()
            self._redis_available = True
            logger.info(
                f"PendingStore '{self.table_name}': Redis connected successfully"
            )
        except ImportError:
            logger.warning(
                "PendingStore: upstash-redis package not installed — "
                "falling back to in-memory + Postgres"
            )
        except Exception as e:
            logger.warning(
                f"PendingStore: Redis connection failed ({e}) — "
                "falling back to in-memory + Postgres"
            )

    # ------------------------------------------------------------------
    # Redis key helpers
    # ------------------------------------------------------------------

    def _redis_key(self, key: str) -> str:
        return f"pending:{self.table_name}:{key}"

    def _redis_ids_key(self) -> str:
        return f"pending:{self.table_name}:ids"

    # ------------------------------------------------------------------
    # Database helpers (Postgres fallback — mirrors original pattern)
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
    # Dict overrides with Redis-first strategy
    # ------------------------------------------------------------------

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        # Always update memory for fast local reads
        super().__setitem__(key, value)

        # Primary: Redis
        if self._redis_available:
            try:
                payload = json.dumps(value, default=str)
                self._redis.setex(self._redis_key(key), _PENDING_TTL, payload)
                self._redis.sadd(self._redis_ids_key(), key)
                logger.debug(
                    f"PendingStore '{self.table_name}': persisted {key[:12]}... in Redis"
                )
                return
            except Exception as e:
                logger.error(f"PendingStore: Redis SET failed — {e}")
                self._redis_available = False  # fall back for subsequent ops

        # Fallback: in-memory + Postgres
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
                        f"PendingStore '{self.table_name}': persisted {key[:12]}... in Postgres"
                    )
            except Exception as e:
                logger.error(f"PendingStore: Postgres persistence failed — {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()

    def __getitem__(self, key: str) -> dict[str, Any]:
        # 1. Check local memory first (fast path)
        try:
            return super().__getitem__(key)
        except KeyError:
            pass

        # 2. Try Redis
        if self._redis_available:
            try:
                raw = self._redis.get(self._redis_key(key))
                if raw is not None:
                    value = json.loads(raw)
                    super().__setitem__(key, value)  # warm local cache
                    return value
            except Exception as e:
                logger.debug(f"PendingStore: Redis GET failed — {e}")

        # 3. Fallback: check Postgres
        conn = self._get_connection()
        if not conn:
            raise KeyError(key)
        try:
            if self._ensure_table(conn):
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT payload FROM {self.table_name} WHERE id = %s", (key,)
                    )
                    row = cur.fetchone()
                if row:
                    value = json.loads(row[0])
                    super().__setitem__(key, value)  # warm local cache
                    return value
        except Exception as e:
            logger.debug(f"PendingStore: Postgres read failed — {e}")
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

        # Remove from Redis
        if self._redis_available:
            try:
                self._redis.delete(self._redis_key(key))
                self._redis.srem(self._redis_ids_key(), key)
                logger.debug(
                    f"PendingStore '{self.table_name}': deleted {key[:12]}... from Redis"
                )
                return
            except Exception as e:
                logger.warning(f"PendingStore: Redis DELETE failed — {e}")
                self._redis_available = False

        # Fallback: remove from Postgres
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM {self.table_name} WHERE id = %s", (key,)
                        )
                    conn.commit()
                    logger.debug(
                        f"PendingStore '{self.table_name}': deleted {key[:12]}... from Postgres"
                    )
            except Exception as e:
                logger.warning(f"PendingStore: Postgres DELETE failed — {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()

    def __contains__(self, key: object) -> bool:
        # Check memory first
        if super().__contains__(key):
            return True
        try:
            self.__getitem__(str(key))
            return True
        except (KeyError, TypeError):
            return False

    def clear(self) -> None:
        """Clear all entries from memory, Redis, and Postgres."""
        # Clear memory
        super().clear()

        # Clear Redis
        if self._redis_available:
            try:
                ids = self._redis.smembers(self._redis_ids_key())
                if ids:
                    for kid in ids:
                        self._redis.delete(self._redis_key(kid))
                    self._redis.delete(self._redis_ids_key())
                    logger.info(
                        f"PendingStore '{self.table_name}': cleared {len(ids)} entries from Redis"
                    )
                return
            except Exception as e:
                logger.warning(f"PendingStore: Redis CLEAR failed — {e}")
                self._redis_available = False

        # Fallback: clear Postgres
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(f"DELETE FROM {self.table_name}")
                    conn.commit()
                    logger.info(
                        f"PendingStore '{self.table_name}': cleared all entries from Postgres"
                    )
            except Exception as e:
                logger.warning(f"PendingStore: Postgres CLEAR failed — {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Bulk read helpers (Redis SMEMBERS + GET)
    # ------------------------------------------------------------------

    def _load_all_from_backend(self) -> None:
        """Hydrate the in-memory cache from Redis (or Postgres fallback)."""
        if self._redis_available:
            try:
                ids = self._redis.smembers(self._redis_ids_key())
                if not ids:
                    return
                for kid in ids:
                    raw = self._redis.get(self._redis_key(kid))
                    if raw is not None:
                        value = json.loads(raw)
                        super().__setitem__(kid, value)
                logger.info(
                    f"PendingStore '{self.table_name}': recovered {len(ids)} entries from Redis"
                )
                return
            except Exception as e:
                logger.warning(f"PendingStore: Redis SMEMBERS failed — {e}")

        # Fallback: Postgres
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
                        f"PendingStore '{self.table_name}': recovered {len(rows)} entries from Postgres"
                    )
        except Exception as e:
            logger.warning(f"PendingStore: Postgres bulk load failed — {e}")
        finally:
            if conn:
                conn.close()

    def items(self):
        if not super().__len__():
            self._load_all_from_backend()
        return super().items()

    def __iter__(self):
        if not super().__len__():
            self._load_all_from_backend()
        return super().__iter__()

    def __len__(self):
        if not super().__len__():
            self._load_all_from_backend()
        return super().__len__()


# ─── Shared module-level singletons ────────────────────────────────────

# Shared pending SQL queries — used by graph nodes AND /sql routes
pending_queries: PendingStore = PendingStore(table_name="idop_pending_queries")

# Shared pending mutation sessions — used by mutation route
pending_mutations: PendingStore = PendingStore(table_name="idop_pending_mutations")


def reset_pending_store() -> None:
    """
    Clear both pending stores and reset Redis availability flags.

    Useful for test isolation to prevent cross-test contamination from
    shared module-level stores.  If Redis is available, keys are cleared
    there too; otherwise the in-memory dict and Postgres fallback are used.
    """
    pending_queries.clear()
    pending_mutations.clear()
