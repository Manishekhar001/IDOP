import secrets
import logging
import psycopg2
from typing import Dict
from app.config import get_settings

logger = logging.getLogger("idop_app.approval_gate")


class ApprovalGate:
    """
    Manages cryptographic session tokens for human-in-the-loop SQL and mutation operations.
    Persisted in PostgreSQL database with an in-memory fallback for local/offline testing.
    """

    def __init__(self):
        # Maps query_id -> session_token (Always kept updated for backward compatibility & tests)
        self.active_sessions: Dict[str, str] = {}

    def _get_connection(self):
        """
        Get connection to the Supabase database (company data) for token persistence.
        Falls back to in-memory if Supabase is unavailable.
        """
        settings = get_settings()
        if not settings.supabase_db_url:
            return None
        try:
            conn = psycopg2.connect(settings.supabase_db_url)
            return conn
        except Exception as e:
            logger.debug(f"Supabase connection failed (falling back to memory): {e}")
            return None

    def _ensure_table(self, conn) -> bool:
        """
        Create the approval tokens table if it does not exist.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS idop_approval_tokens (
            query_id VARCHAR(100) PRIMARY KEY,
            token VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Could not create approval tokens table: {e}")
            conn.rollback()
            return False

    def generate_session(self, query_id: str) -> str:
        """
        Generate a secure, single-use approval token. Writes to Postgres and updates memory.
        """
        token = secrets.token_hex(32)

        # Always update memory for tests/fallback
        self.active_sessions[query_id] = token

        # Try to persist in database
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO idop_approval_tokens (query_id, token)
                            VALUES (%s, %s)
                            ON CONFLICT (query_id) 
                            DO UPDATE SET token = EXCLUDED.token
                            """,
                            (query_id, token),
                        )
                    conn.commit()
                    logger.info(
                        f"Persisted approval session token for ID {query_id} in PostgreSQL"
                    )
            except Exception as e:
                logger.error(f"Failed to persist approval token in PostgreSQL: {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()
        else:
            logger.info(f"Generated ephemeral approval session for ID {query_id}")

        return token

    def verify_and_close_session(self, query_id: str, token: str) -> bool:
        """
        Validate the approval token and remove it if matches (one-time use).
        """
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    # Fetch from database
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT token FROM idop_approval_tokens WHERE query_id = %s",
                            (query_id,),
                        )
                        row = cur.fetchone()

                    if not row:
                        logger.warning(
                            f"Verification failed: Query ID {query_id} not found in database"
                        )
                        # Synchronize memory state if database is source of truth
                        if query_id in self.active_sessions:
                            del self.active_sessions[query_id]
                        conn.close()
                        return False

                    stored_token = row[0]
                    if secrets.compare_digest(stored_token, token):
                        # Delete from database
                        with conn.cursor() as cur:
                            cur.execute(
                                "DELETE FROM idop_approval_tokens WHERE query_id = %s",
                                (query_id,),
                            )
                        conn.commit()

                        # Sync memory
                        if query_id in self.active_sessions:
                            del self.active_sessions[query_id]

                        logger.info(
                            f"✓ Verification success: Database session closed for query {query_id}"
                        )
                        conn.close()
                        return True

                    logger.warning(
                        f"Verification failed: Incorrect token in DB for query {query_id}"
                    )
                    conn.close()
                    return False
            except Exception as e:
                logger.error(
                    f"Database token verification failed: {e}. Falling back to memory validation."
                )
                if conn:
                    conn.rollback()
                    conn.close()
            finally:
                if conn and not conn.closed:
                    conn.close()

        # Ephemeral memory fallback (used in unit tests / database outages)
        if query_id not in self.active_sessions:
            logger.warning(
                f"Verification failed (Memory Fallback): Query ID {query_id} not found"
            )
            return False

        stored_token = self.active_sessions[query_id]
        if secrets.compare_digest(stored_token, token):
            del self.active_sessions[query_id]
            logger.info(
                f"✓ Verification success (Memory Fallback): Session closed for query {query_id}"
            )
            return True

        logger.warning(
            f"Verification failed (Memory Fallback): Incorrect token for query {query_id}"
        )
        return False


# Shared singleton instance for cross-module access
approval_gate = ApprovalGate()
