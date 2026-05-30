import secrets
import logging
import psycopg2
from typing import Dict
from app.config import get_settings

logger = logging.getLogger("idop_app.mutation_approval_gate")


class MutationApprovalGate:
    """
    Manages session validation and cryptographic tokens for bulk mutation approvals.
    Persisted in PostgreSQL database with an in-memory fallback for local/offline testing.
    """

    def __init__(self):
        # Maps mutation_id -> session_token (Always kept updated for backward compatibility & tests)
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
        Create the mutation approval tokens table if it does not exist.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS idop_mutation_approval_tokens (
            mutation_id VARCHAR(100) PRIMARY KEY,
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
            logger.warning(f"Could not create mutation approval tokens table: {e}")
            conn.rollback()
            return False

    def generate_session(self, mutation_id: str) -> str:
        """
        Generate a secure, single-use approval token. Writes to Postgres and updates memory.
        """
        token = secrets.token_hex(32)
        
        # Always update memory for tests/fallback
        self.active_sessions[mutation_id] = token

        # Try to persist in database
        conn = self._get_connection()
        if conn:
            try:
                if self._ensure_table(conn):
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO idop_mutation_approval_tokens (mutation_id, token)
                            VALUES (%s, %s)
                            ON CONFLICT (mutation_id) 
                            DO UPDATE SET token = EXCLUDED.token
                            """,
                            (mutation_id, token)
                        )
                    conn.commit()
                    logger.info(f"Persisted mutation approval session token for ID {mutation_id} in PostgreSQL")
            except Exception as e:
                logger.error(f"Failed to persist mutation approval token in PostgreSQL: {e}")
                if conn:
                    conn.rollback()
            finally:
                conn.close()
        else:
            logger.info(f"Generated ephemeral mutation approval session for ID {mutation_id}")

        return token

    def verify_and_close_session(self, mutation_id: str, token: str) -> bool:
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
                            "SELECT token FROM idop_mutation_approval_tokens WHERE mutation_id = %s",
                            (mutation_id,)
                        )
                        row = cur.fetchone()
                    
                    if not row:
                        logger.warning(f"Verification failed: Mutation ID {mutation_id} not found in database")
                        # Synchronize memory state if database is source of truth
                        if mutation_id in self.active_sessions:
                            del self.active_sessions[mutation_id]
                        conn.close()
                        return False

                    stored_token = row[0]
                    if secrets.compare_digest(stored_token, token):
                        # Delete from database
                        with conn.cursor() as cur:
                            cur.execute(
                                "DELETE FROM idop_mutation_approval_tokens WHERE mutation_id = %s",
                                (mutation_id,)
                            )
                        conn.commit()
                        
                        # Sync memory
                        if mutation_id in self.active_sessions:
                            del self.active_sessions[mutation_id]
                            
                        logger.info(f"✓ Verification success: Database mutation session closed for ID {mutation_id}")
                        conn.close()
                        return True

                    logger.warning(f"Verification failed: Incorrect token in DB for Mutation ID {mutation_id}")
                    conn.close()
                    return False
            except Exception as e:
                logger.error(f"Database mutation token verification failed: {e}. Falling back to memory validation.")
                if conn:
                    conn.rollback()
                    conn.close()
            finally:
                if conn and not conn.closed:
                    conn.close()

        # Ephemeral memory fallback (used in unit tests / database outages)
        if mutation_id not in self.active_sessions:
            logger.warning(f"Verification failed (Memory Fallback): Mutation ID {mutation_id} not found")
            return False

        stored_token = self.active_sessions[mutation_id]
        if secrets.compare_digest(stored_token, token):
            del self.active_sessions[mutation_id]
            logger.info(f"✓ Verification success (Memory Fallback): Mutation session closed for ID {mutation_id}")
            return True

        logger.warning(f"Verification failed (Memory Fallback): Incorrect token for Mutation ID {mutation_id}")
        return False
