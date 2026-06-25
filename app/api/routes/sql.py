import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import get_current_user
from app.api.schemas import (
    ErrorResponse,
    SQLApprovalRequest,
    SQLExecuteResponse,
    SQLGenerationRequest,
    SQLResponse,
)
from app.core.approval_gate import approval_gate as gate
from app.core.feature1_sql.executor import SQLExecutor
from app.core.feature1_sql.shared import sql_service
from app.opik import track
from app.services.pending_store import pending_queries as shared_pending_queries
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/sql", tags=["SQL Operations"])

# Use centralized singleton for shared service, cache stats
executor = SQLExecutor()


@router.post(
    "/generate",
    response_model=SQLResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question or parameters"},
        500: {"model": ErrorResponse, "description": "LLM or cache error"},
    },
    summary="Generate SQL for approval",
    description=(
        "Converts a natural language question into a SQL query using the Vanna agent. "
        "Returns the generated SQL with a judge explanation and a cryptographic approval token. "
        "The query is stored in a pending queue and must be approved via POST /sql/approve before execution."
    ),
)
@track(name="generate_sql")
async def generate_sql(
    body: SQLGenerationRequest,
    _user: dict = Depends(get_current_user),
) -> SQLResponse:
    """
    Generate a SQL query from natural language using Vanna and cache-aware generation.

    Pipeline: validate → check cache → generate SQL via Vanna → LLM judge audit → store pending → return token

    Args:
        body: The generation request containing the natural language question, temperature,
              seed, and top-p sampling parameters for the Vanna agent.

    Returns:
        SQLResponse: Generated SQL query with judge explanation, status, and approval token.

    Raises:
        HTTPException 400: If the question is invalid or parameters are out of range.
        HTTPException 500: If LLM generation, cache, or database operations fail.
    """
    logger.info(
        f"SQL generation request: {body.question} (temp={body.vanna_temperature})"
    )
    try:
        res = await sql_service.generate_sql_for_approval(
            question=body.question,
            explain=body.explain,
            vanna_temperature=body.vanna_temperature,
            vanna_seed=body.vanna_seed,
            vanna_top_p=body.vanna_top_p,
        )

        # Crypto gate generation (synchronous psycopg2 — offload to thread)
        token = await asyncio.to_thread(gate.generate_session, res["query_id"])

        # Store in shared pending store so graph nodes and routes share the same data
        query_id = res["query_id"]
        shared_pending_queries[query_id] = {
            "question": res["question"],
            "sql": res["sql"],
            "status": "pending_approval",
            "token": token,
        }

        return SQLResponse(
            query_id=res["query_id"],
            question=res["question"],
            sql=res["sql"],
            explanation=res["explanation"],
            status=res["status"],
            cache_hit=res["cache_hit"],
            token=token,
        )
    except Exception as e:
        logger.error(f"SQL generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/approve",
    response_model=SQLExecuteResponse,
    responses={
        403: {
            "model": ErrorResponse,
            "description": "Invalid or expired approval token",
        },
        404: {"model": ErrorResponse, "description": "Query session not found"},
        500: {"model": ErrorResponse, "description": "Database execution error"},
    },
    summary="Approve and execute SQL",
    description=(
        "Approve or reject a pending SQL query. Requires the cryptographic token "
        "returned by POST /sql/generate. On approval, executes the query against "
        "PostgreSQL and returns the results. On rejection, removes the query from the pending queue."
    ),
)
@track(name="approve_sql")
async def approve_sql(
    body: SQLApprovalRequest,
    _user: dict = Depends(get_current_user),
) -> SQLExecuteResponse:
    """
    Approve or reject a pending SQL query with cryptographic token verification.

    Pipeline: verify token → handle rejection → execute SQL → return results

    Args:
        body: The approval request containing query_id, approved flag, and cryptographic token.

    Returns:
        SQLExecuteResponse: Execution results with row count, status, and optional cache hit indicator.

    Raises:
        HTTPException 403: If the cryptographic token is invalid, expired, or already used.
        HTTPException 404: If the query session ID is not found in the pending register.
        HTTPException 500: If the database execution fails.
    """
    logger.info(
        f"SQL approval request for Query ID: {body.query_id}, Approved: {body.approved}"
    )

    # 1. Verify Cryptographic Token
    if body.approved:
        if not await asyncio.to_thread(
            gate.verify_and_close_session, body.query_id, body.token
        ):
            raise HTTPException(
                status_code=403,
                detail="Invalid, expired or already used cryptographic approval token.",
            )

    # 2. Handle Rejection
    if not body.approved:
        if body.query_id in shared_pending_queries:
            del shared_pending_queries[body.query_id]
        return SQLExecuteResponse(
            query_id=body.query_id,
            sql="",
            results=[],
            result_count=0,
            status="rejected",
        )

    # 3. Handle Execution
    if body.query_id not in shared_pending_queries:
        raise HTTPException(
            status_code=404, detail="Query session not found in pending register."
        )

    query_info = shared_pending_queries[body.query_id]
    sql = query_info["sql"]
    question = query_info["question"]

    try:
        # Run standard execute and log
        results = await asyncio.to_thread(
            executor.execute_and_log, body.query_id, question, sql
        )

        # Remove from pending queue
        if body.query_id in shared_pending_queries:
            del shared_pending_queries[body.query_id]

        return SQLExecuteResponse(
            query_id=body.query_id,
            sql=sql,
            results=results,
            result_count=len(results),
            status="executed",
        )
    except Exception as e:
        logger.error(f"SQL execution endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database execution failed: {e!s}")


@router.get(
    "/pending",
    summary="Get all pending SQL queries",
    description="Returns all SQL queries awaiting human approval. Each entry includes the query ID, original question, SQL statement, and approval status.",
)
@track(name="get_pending_sql")
async def get_pending(
    _user: dict = Depends(get_current_user),
) -> list[dict]:
    """
    Retrieve all pending SQL queries awaiting human approval.

    Returns a list of pending queries with their session IDs, original questions,
    generated SQL statements, and current status. Use POST /sql/approve to
    approve or reject a specific query.

    Returns:
        list[dict]: List of pending query objects with query_id, question, sql, and status fields.
    """
    return [{"query_id": qid, **info} for qid, info in shared_pending_queries.items()]
