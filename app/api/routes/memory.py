"""
Memory management routes for Long-Term Memory (LTM) personalization.

Provides endpoints to retrieve and delete user-specific memory facts
stored in the Postgres-backed AsyncPostgresStore. Memory facts are
extracted from conversations and persist across sessions for personalized
interactions.
"""

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    DeleteMemoryResponse,
    ErrorResponse,
    MemoryItem,
    MemoryListResponse,
)
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


def get_store(request: Request):
    """Dependency: retrieve the AsyncPostgresStore from app state."""
    return request.app.state.store


@router.get(
    "/{user_id}",
    response_model=MemoryListResponse,
    responses={500: {"model": ErrorResponse, "description": "Postgres query error"}},
    summary="List user memories",
    description=(
        "Return all long-term memory (LTM) facts stored for a given user. "
        "These facts are extracted from conversation history and persist across "
        "sessions to provide personalized context in future interactions."
    ),
)
@track(name="list_memories")
async def list_memories(user_id: str, request: Request) -> MemoryListResponse:
    """
    Retrieve all long-term memory facts for a specific user.

    Queries the Postgres-backed AsyncPostgresStore for memory items stored under
    the namespace (user, user_id, details). Each item represents a persistent
    profile fact extracted from conversation history.

    Args:
        user_id: The unique user identifier to retrieve memories for.

    Returns:
        MemoryListResponse: List of memory facts with their content, user ID, and count.

    Raises:
        HTTPException 500: If the Postgres query fails or the store is unavailable.
    """
    logger.debug(f"Listing memories for user={user_id}")
    store = get_store(request)

    try:
        ns = ("user", user_id, "details")
        items = await store.asearch(ns)
        memories = [MemoryItem(data=it.value.get("data", "")) for it in items]
        return MemoryListResponse(
            user_id=user_id, memories=memories, count=len(memories)
        )
    except Exception as e:
        logger.error(f"list_memories error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve memories: {e!s}"
        )


@router.delete(
    "/{user_id}",
    response_model=DeleteMemoryResponse,
    responses={500: {"model": ErrorResponse, "description": "Postgres deletion error"}},
    summary="Clear user memories",
    description=(
        "Delete all long-term memory facts for a given user. "
        "This operation is irreversible — removed facts cannot be recovered. "
        "The system will re-extract facts from future conversations."
    ),
)
@track(name="delete_memories")
async def delete_memories(user_id: str, request: Request) -> DeleteMemoryResponse:
    """
    Delete all long-term memory facts for a specific user.

    Iterates through all memory items stored under the namespace
    (user, user_id, details) and deletes each one from the Postgres store.
    This operation is irreversible.

    Args:
        user_id: The unique user identifier whose memories should be deleted.

    Returns:
        DeleteMemoryResponse: Confirmation message with the count of deleted memories.

    Raises:
        HTTPException 500: If the Postgres deletion fails or the store is unavailable.
    """
    logger.warning(f"Deleting all memories for user={user_id}")
    store = get_store(request)

    try:
        ns = ("user", user_id, "details")
        items = await store.asearch(ns)
        for item in items:
            await store.adelete(ns, item.key)

        logger.info(f"Deleted {len(items)} memories for user={user_id}")
        return DeleteMemoryResponse(
            message=f"Deleted {len(items)} memories successfully", user_id=user_id
        )
    except Exception as e:
        logger.error(f"delete_memories error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memories: {e!s}")
