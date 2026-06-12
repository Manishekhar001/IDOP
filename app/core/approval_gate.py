"""
Shared Approval Gate — parameterized cryptographic token manager.

Consolidates the near-identical ``ApprovalGate`` (Feature 1 / SQL) and
``MutationApprovalGate`` (Feature 2 / mutations) into a single class.
Both modules import from here with different table names and column names.

Usage:

    # Feature 1 — SQL approvals
    from app.core.approval_gate import approval_gate as gate

    token = gate.generate_session("query-uuid")
    gate.verify_and_close_session("query-uuid", token)

    # Feature 2 — mutation approvals
    from app.core.approval_gate import mutation_approval_gate as gate

    token = gate.generate_session("mutation-uuid")
    gate.verify_and_close_session("mutation-uuid", token)
"""

import secrets
import logging
import psycopg2
from typing import Dict
from app.config import get_settings


class ApprovalGate:
    """
    Manages cryptographic session tokens for human-in-the-loop operations
    (SQL queries and database mutations).

    Tokens are persisted in a PostgreSQL table (configurable via the *table_name*
    constructor argument) with an in-memory fallback for local / offline testing.
    Each token is single-use — after ``verify_and_close_session`` it is deleted
    from both the database and the in-memory cache.
    """

    def __init__(self, table_name: str, session_column: str, logger_name: str):
        """
        Args:
            table_name: The PostgreSQL table used to persist tokens
                        (e.g. ``"idop_approval_tokens"`` or
                         ``"idop_mutation_approval_tokens"``).
            session_column: The primary key column name for the session ID
                            (e.g. ``"query_id"`` for SQL or ``"mutation_id"``
                             for mutations).
            logger_name: Name for the module-level logger
                         (e.g. ``"idop_app.approval_gate"``).
        """
        self.table_name = table_name
        self.session_column = session_column
        self.logger = logging.getLogger(logger_name)
        # Maps session_id -> session_token (always kept up to date for fast lookups)
        self.active_sessions: Dict[str, str] = {}

    def _get_connection(self):
        """Get connection to the Supabase database for token persistence."""
        settings = get_settings()
        if not settings.supabase_db_url:
            return None
        try:
            conn = psycopg2.connect(settings.supabase_db_url)
            return conn
        except Exception as e:
            self.logger.debug(
                f"Supabase connection failed (falling back to memory): {e}"
            )
            return None

    def _ensure_table(self, conn) -> bool:
        """
        Create the approval tokens table if it does not exist.

        Uses ``self.session_column`` as the primary key column name to
        match the original schema (``query_id`` for SQL, ``mutation_id``
        for mutations).
        """
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.session_column} VARCHAR(100) PRIMARY KEY,
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
            self.logger.warning(f"Could not create {self.table_name} table: {e}")
            conn.rollback()
            return False

    def generate_session(self, session_id: str) -> str:
        """
        Generate a secure, single-use approval token.

        Persists to the PostgreSQL table (``self.table_name``) and always
        updates the in-memory cache for fast lookups.
        """
        token = secrets.token_hex(32)

        # Always update memory for tests / fallback
        self.active_sessions[session_id] = token

        # Try to persist in database
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            INSERT INTO {self.table_name} ({self.session_column}, token)
                            VALUES (%s, %s)
                            ON CONFLICT ({self.session_column})
                            DO UPDATE SET token = EXCLUDED.token
                            """,
                            (session_id, token),
                        )
                    conn.commit()
                    self.logger.info(
                        f"Persisted approval session token for ID {session_id} in PostgreSQL"
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to persist approval token in PostgreSQL: {e}"
                )
                conn.rollback()
            finally:
                conn.close()
        else:
            self.logger.info(
                f"Generated ephemeral approval session for ID {session_id}"
            )

        return token

    def verify_and_close_session(self, session_id: str, token: str) -> bool:
        """
        Validate the approval token and remove it if it matches (single-use).

        Returns True if the token is valid and was consumed. False otherwise.
        """
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(
                            f"SELECT token FROM {self.table_name} WHERE {self.session_column} = %s",
                            (session_id,),
                        )
                        row = cur.fetchone()

                    if not row:
                        self.logger.warning(
                            f"Verification failed: ID {session_id} not found in database"
                        )
                        # Synchronize memory state if database is source of truth
                        if session_id in self.active_sessions:
                            del self.active_sessions[session_id]
                        return False

                    stored_token = row[0]
                    if secrets.compare_digest(stored_token, token):
                        with conn.cursor() as cur:
                            cur.execute(
                                f"DELETE FROM {self.table_name} WHERE {self.session_column} = %s",
                                (session_id,),
                            )
                        conn.commit()

                        # Sync memory
                        if session_id in self.active_sessions:
                            del self.active_sessions[session_id]

                        self.logger.info(
                            f"✓ Verification success: Database session closed for ID {session_id}"
                        )
                        return True

                    self.logger.warning(
                        f"Verification failed: Incorrect token in DB for ID {session_id}"
                    )
                    return False
            except Exception as e:
                self.logger.error(
                    f"Database token verification failed: {e}. Falling back to memory validation."
                )
                conn.rollback()
            finally:
                if conn and not conn.closed:
                    conn.close()

        # Ephemeral memory fallback (used in unit tests / database outages)
        if session_id not in self.active_sessions:
            self.logger.warning(
                f"Verification failed (Memory Fallback): ID {session_id} not found"
            )
            return False

        stored_token = self.active_sessions[session_id]
        if secrets.compare_digest(stored_token, token):
            del self.active_sessions[session_id]
            self.logger.info(
                f"✓ Verification success (Memory Fallback): Session closed for ID {session_id}"
            )
            return True

        self.logger.warning(
            f"Verification failed (Memory Fallback): Incorrect token for ID {session_id}"
        )
        return False


# Shared singleton instances for cross-module access
approval_gate = ApprovalGate(
    table_name="idop_approval_tokens",
    session_column="query_id",
    logger_name="idop_app.approval_gate",
)

mutation_approval_gate = ApprovalGate(
    table_name="idop_mutation_approval_tokens",
    session_column="mutation_id",
    logger_name="idop_app.mutation_approval_gate",
)
