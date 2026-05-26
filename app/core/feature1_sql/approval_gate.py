import secrets
import logging
from typing import Dict, Any

logger = logging.getLogger("idop_app.approval_gate")


class ApprovalGate:
    """
    Manages cryptographic session tokens for human-in-the-loop SQL and mutation operations.
    """

    def __init__(self):
        # Maps query_id -> session_token
        self.active_sessions: Dict[str, str] = {}

    def generate_session(self, query_id: str) -> str:
        """
        Generate a secure, single-use approval token.
        """
        token = secrets.token_hex(32)
        self.active_sessions[query_id] = token
        logger.info(f"Generated approval session for ID {query_id}")
        return token

    def verify_and_close_session(self, query_id: str, token: str) -> bool:
        """
        Validate the approval token and remove it if matches (one-time use).
        """
        if query_id not in self.active_sessions:
            logger.warning(f"Verification failed: Query ID {query_id} not found")
            return False

        stored_token = self.active_sessions[query_id]
        if secrets.compare_digest(stored_token, token):
            del self.active_sessions[query_id]
            logger.info(f"✓ Verification success: Session closed for query {query_id}")
            return True

        logger.warning(f"Verification failed: Incorrect token for query {query_id}")
        return False


# Shared singleton instance for cross-module access
approval_gate = ApprovalGate()

