from fastapi import APIRouter, HTTPException
from app.api.schemas import SQLApprovalRequest, SQLResponse, SQLExecuteResponse, ErrorResponse, SQLGenerationRequest
from app.core.feature1_sql.vanna_service import TextToSQLService
from app.core.feature1_sql.approval_gate import approval_gate as gate
from app.core.feature1_sql.executor import SQLExecutor
from app.services.query_cache_service import QueryCacheService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/sql", tags=["SQL Operations"])

# Shared services
query_cache = QueryCacheService()
sql_service = TextToSQLService(query_cache_service=query_cache)
executor = SQLExecutor()


@router.post(
    "/generate",
    response_model=SQLResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Generate SQL for approval",
)
async def generate_sql(body: SQLGenerationRequest) -> SQLResponse:
    logger.info(f"SQL generation request: {body.question} (temp={body.vanna_temperature})")
    try:
        res = await sql_service.generate_sql_for_approval(
            question=body.question,
            explain=body.explain,
            vanna_temperature=body.vanna_temperature,
            vanna_seed=body.vanna_seed,
            vanna_top_p=body.vanna_top_p,
        )
        
        # Crypto gate generation
        token = gate.generate_session(res["query_id"])
        
        # Update the pending register with the token
        query_id = res["query_id"]
        if query_id in sql_service.pending_queries:
            sql_service.pending_queries[query_id]["token"] = token
            
        return SQLResponse(
            query_id=res["query_id"],
            question=res["question"],
            sql=res["sql"],
            explanation=res["explanation"],
            status=res["status"],
            cache_hit=res["cache_hit"],
            token=token
        )
    except Exception as e:
        logger.error(f"SQL generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/approve",
    response_model=SQLExecuteResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Approve and execute SQL",
)
async def approve_sql(body: SQLApprovalRequest) -> SQLExecuteResponse:
    logger.info(f"SQL approval request for Query ID: {body.query_id}, Approved: {body.approved}")
    
    # 1. Verify Cryptographic Token
    if body.approved:
        if not gate.verify_and_close_session(body.query_id, body.token):
            raise HTTPException(status_code=403, detail="Invalid, expired or already used cryptographic approval token.")

    # 2. Handle Rejection
    if not body.approved:
        if body.query_id in sql_service.pending_queries:
            del sql_service.pending_queries[body.query_id]
        return SQLExecuteResponse(
            query_id=body.query_id,
            sql="",
            results=[],
            result_count=0,
            status="rejected"
        )

    # 3. Handle Execution
    if body.query_id not in sql_service.pending_queries:
        raise HTTPException(status_code=404, detail="Query session not found in pending register.")

    query_info = sql_service.pending_queries[body.query_id]
    sql = query_info["sql"]
    question = query_info["question"]

    try:
        # Run standard execute and log
        results = executor.execute_and_log(body.query_id, question, sql)
        
        # Remove from pending queue
        if body.query_id in sql_service.pending_queries:
            del sql_service.pending_queries[body.query_id]

        return SQLExecuteResponse(
            query_id=body.query_id,
            sql=sql,
            results=results,
            result_count=len(results),
            status="executed"
        )
    except Exception as e:
        logger.error(f"SQL execution endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database execution failed: {str(e)}")


@router.get(
    "/pending",
    summary="Get all pending SQL queries",
)
async def get_pending():
    return sql_service.get_pending_queries()
