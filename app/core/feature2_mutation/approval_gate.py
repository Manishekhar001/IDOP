import secrets
import logging
from typing import Dict

logger = logging.getLogger("idop_app.mutation_approval_gate")


class MutationApprovalGate:
    """
    Manages session validation and cryptographic tokens for bulk mutation approvals.
    """

    def __init__(self):
        self.active_sessions: Dict[str, str] = {}

    def generate_session(self, mutation_id: str) -> str:
        token = secrets.token_hex(32)
        self.active_sessions[mutation_id] = token
        logger.info(f"Generated mutation approval session for ID {mutation_id}")
        return token

    def verify_and_close_session(self, mutation_id: str, token: str) -> bool:
        if mutation_id not in self.active_sessions:
            logger.warning(f"Verification failed: Mutation ID {mutation_id} not found")
            return False

        stored_token = self.active_sessions[mutation_id]
        if secrets.compare_digest(stored_token, token):
            del self.active_sessions[mutation_id]
            logger.info(f"✓ Verification success: Mutation session closed for ID {mutation_id}")
            return True

        logger.warning(f"Verification failed: Incorrect token for Mutation ID {mutation_id}")
        return False
