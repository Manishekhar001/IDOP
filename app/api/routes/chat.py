import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.api.schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SourceDocument,
)
from app.core.csrag_engine import CSRAGEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


def get_engine(request: Request) -> CSRAGEngine:
    return request.app.state.engine


def get_checkpointer(request: Request):
    return request.app.state.checkpointer


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Ask a question",
)
async def chat(
    body: ChatRequest,
    engine: CSRAGEngine = Depends(get_engine),
) -> ChatResponse:
    logger.info(
        f"Chat — thread={body.thread_id}, user={body.user_id}, "
        f"q='{body.question[:80]}'"
    )
    start_time = time.time()

    try:
        result = await engine.aquery(
            question=body.question,
            thread_id=body.thread_id,
            user_id=body.user_id,
            search_mode=body.search_mode,
            top_k=body.top_k,
            enable_hyde=body.enable_hyde,
            enable_reranking=body.enable_reranking,
        )
    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {type(e).__name__}: {str(e)}",
        )

    processing_time = (time.time() - start_time) * 1000

    sources: list[SourceDocument] | None = None
    if body.include_sources:
        sources = [
            SourceDocument(
                content=s["content"],
                metadata=s["metadata"],
                origin=s["origin"],
            )
            for s in result.get("sources", [])
        ]

    # Support NL-to-SQL or Mutation response interception:
    # If the router directed to SQL, return information containing SQL status details
    answer_text = result["answer"]
    if result.get("query_type") == "SQL":
        answer_text = f"Generated approved SQL Session: {result.get('sql_query_id')}\nQuery: {result.get('sql_query')}\nStatus: {result.get('sql_status')}\nToken: {result.get('approval_token')}"

    return ChatResponse(
        question=body.question,
        answer=answer_text,
        sources=sources,
        processing_time_ms=round(processing_time, 2),
        crag_verdict=result.get("crag_verdict", ""),
        crag_reason=result.get("crag_reason", ""),
        issup=result.get("issup", ""),
        evidence=result.get("evidence", []),
        isuse=result.get("isuse", ""),
        use_reason=result.get("use_reason", ""),
        retries=result.get("retries", 0),
        rewrite_tries=result.get("rewrite_tries", 0),
        sql_query=result.get("sql_query") if result.get("sql_query") else None,
        sql_results=result.get("sql_results") if result.get("sql_results") else None,
        hyde_used=result.get("hyde_used", False),
        hyde_hypotheses=result.get("hyde_hypotheses") if result.get("hyde_hypotheses") else None,
        reranking_used=result.get("reranking_used", False),
        
        # New rich operational detail fields
        query_type=result.get("query_type") if result.get("query_type") else None,
        ltm_context=result.get("ltm_context") if result.get("ltm_context") else None,
        mutation_id=result.get("mutation_id") if result.get("mutation_id") else None,
        mutation_table=result.get("mutation_table") if result.get("mutation_table") else None,
        mutation_op=result.get("mutation_op") if result.get("mutation_op") else None,
        mutation_status=result.get("mutation_status") if result.get("mutation_status") else None,
        mutation_error=result.get("mutation_error") if result.get("mutation_error") else None,
        mutation_result_count=result.get("mutation_result_count") if result.get("mutation_result_count") else None,
        approval_token=result.get("approval_token") if result.get("approval_token") else None,
    )


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Streaming error"},
    },
    summary="Ask a question (streaming)",
)
async def chat_stream(
    body: ChatRequest,
    engine: CSRAGEngine = Depends(get_engine),
) -> StreamingResponse:
    logger.info(
        f"Chat stream — thread={body.thread_id}, user={body.user_id}, "
        f"q='{body.question[:80]}'"
    )

    async def generate():
        try:
            async for chunk in engine.astream(
                question=body.question,
                thread_id=body.thread_id,
                user_id=body.user_id,
                search_mode=body.search_mode,
                top_k=body.top_k,
                enable_hyde=body.enable_hyde,
                enable_reranking=body.enable_reranking,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"\n\n[Error: {type(e).__name__}: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@router.get(
    "/history/{thread_id}",
    response_model=ChatHistoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
        500: {"model": ErrorResponse, "description": "Retrieval error"},
    },
    summary="Get conversation history",
)
async def get_chat_history(
    thread_id: str,
    request: Request,
) -> ChatHistoryResponse:
    logger.info(f"History request — thread={thread_id}")
    checkpointer = get_checkpointer(request)

    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception as e:
        logger.error(f"History retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {type(e).__name__}: {str(e)}",
        )

    if checkpoint_tuple is None:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found for thread_id='{thread_id}'",
        )

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    raw_messages = channel_values.get("messages", [])
    summary = channel_values.get("summary", "")

    messages: list[ChatMessage] = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            messages.append(ChatMessage(role="human", content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(ChatMessage(role="assistant", content=msg.content))
        elif isinstance(msg, dict):
            msg_type = msg.get("type", "").lower()
            content = msg.get("content", "")
            if msg_type == "human":
                messages.append(ChatMessage(role="human", content=content))
            elif msg_type == "ai":
                messages.append(ChatMessage(role="assistant", content=content))

    logger.info(
        f"History returned — thread={thread_id}, "
        f"messages={len(messages)}, summary={'yes' if summary else 'no'}"
    )

    return ChatHistoryResponse(
        thread_id=thread_id,
        messages=messages,
        summary=summary,
        message_count=len(messages),
    )
