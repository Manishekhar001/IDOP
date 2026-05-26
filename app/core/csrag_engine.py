from langchain_core.messages import HumanMessage
import uuid
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.config import get_settings
from app.core.graph.builder import build_graph
from app.core.vector_store import VectorStoreService
from app.services.query_cache_service import QueryCacheService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
query_cache = QueryCacheService()

_RECURSION_LIMIT = 80


class CSRAGEngine:
    def __init__(
        self,
        vector_store: VectorStoreService,
        store: AsyncPostgresStore,
        checkpointer: AsyncPostgresSaver,
    ) -> None:
        self._graph = build_graph(
            vector_store=vector_store,
            store=store,
            checkpointer=checkpointer,
        )
        logger.info("CSRAGEngine initialized with IDOP 4-path router")

    def _build_config(self, thread_id: str, user_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": _RECURSION_LIMIT,
        }

    @staticmethod
    def _initial_state(
        question: str,
        search_mode: str = "hybrid",
        top_k: int = 4,
        enable_hyde: bool = False,
        enable_reranking: bool = False,
    ) -> dict:
        return {
            "messages": [HumanMessage(content=question, id=str(uuid.uuid4()))],
            "summary": "",
            "user_id": "",
            "ltm_context": "",
            "need_retrieval": False,
            "question": "",
            "retrieval_query": "",
            "rewrite_tries": 0,
            "docs": [],
            "good_docs": [],
            "crag_verdict": "",
            "crag_reason": "",
            "web_query": "",
            "web_docs": [],
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "answer": "",
            "issup": "",
            "evidence": [],
            "retries": 0,
            "isuse": "",
            "use_reason": "",
            
            # Advanced Corrective RAG Configs
            "search_mode": search_mode,
            "top_k": top_k,
            "enable_hyde": enable_hyde,
            "enable_reranking": enable_reranking,
            "hyde_used": False,
            "hyde_hypotheses": [],
            "reranking_used": False,
            
            # IDOP State variables
            "query_type": "",
            "sql_query": "",
            "sql_results": [],
            "sql_query_id": "",
            "sql_explanation": "",
            "sql_status": "",
            "mutation_id": "",
            "mutation_table": "",
            "mutation_op": "",
            "mutation_rows": [],
            "mutation_mapped_rows": [],
            "mutation_status": "",
            "mutation_error": "",
            "mutation_result_count": 0,
            "approval_token": ""
        }

    async def aquery(
        self,
        question: str,
        thread_id: str,
        user_id: str,
        search_mode: str = "hybrid",
        top_k: int = 4,
        enable_hyde: bool = False,
        enable_reranking: bool = False,
    ) -> dict:
        logger.info(
            f"async query — thread={thread_id}, user={user_id}, "
            f"q='{question[:80]}'"
        )
        
        cache_key = query_cache.get_rag_key(question, top_k)
        if (query_cache.enabled or query_cache.use_local) and search_mode == "hybrid" and not enable_hyde:
            cached_result = query_cache.get(cache_key, cache_type="rag")
            if cached_result:
                logger.info(f"RAG Cache HIT for: '{question[:50]}'")
                return {**cached_result, "cache_hit": True, "cost_saved": "$0.05"}

        config = self._build_config(thread_id, user_id)
        init_state = self._initial_state(
            question=question,
            search_mode=search_mode,
            top_k=top_k,
            enable_hyde=enable_hyde,
            enable_reranking=enable_reranking
        )
        result = await self._graph.ainvoke(init_state, config)
        formatted = self._format_result(result)

        # Check post-verification gates before writing to Redis/local cache
        if (query_cache.enabled or query_cache.use_local) and formatted.get("query_type") == "RAG":
            crag_verdict = formatted.get("crag_verdict")
            issup = formatted.get("issup")
            isuse = formatted.get("isuse")
            
            if crag_verdict == "CORRECT" and issup == "fully_supported" and isuse == "useful":
                query_cache.set(cache_key, formatted, ttl=settings.cache_ttl_rag, cache_type="rag")
                logger.info("✓ RAG Cache MISS - passed 3-tier quality gates and cached successfully.")
            else:
                logger.info(f"✗ RAG Cache write skipped - failed quality gates (crag={crag_verdict}, sup={issup}, use={isuse})")

        return formatted

    async def astream(
        self,
        question: str,
        thread_id: str,
        user_id: str,
        search_mode: str = "hybrid",
        top_k: int = 4,
        enable_hyde: bool = False,
        enable_reranking: bool = False,
    ):
        logger.info(
            f"streaming query — thread={thread_id}, user={user_id}, "
            f"q='{question[:80]}'"
        )
        config = self._build_config(thread_id, user_id)
        init_state = self._initial_state(
            question=question,
            search_mode=search_mode,
            top_k=top_k,
            enable_hyde=enable_hyde,
            enable_reranking=enable_reranking
        )

        _ANSWER_NODES = {"generate_answer", "generate_direct"}

        try:
            async for msg, metadata in self._graph.astream(
                init_state,
                config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node in _ANSWER_NODES and hasattr(msg, "content") and msg.content:
                    yield msg.content
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"\n\n[Error: {type(e).__name__}: {str(e)}]"

    def health_check(self) -> bool:
        return self._graph is not None

    @staticmethod
    def _format_result(state: dict) -> dict:
        good_docs = state.get("good_docs", []) or []
        web_docs = state.get("web_docs", []) or []

        sources = [
            {
                "content": (
                    d.page_content[:500] + "..."
                    if len(d.page_content) > 500
                    else d.page_content
                ),
                "metadata": d.metadata,
                "origin": "internal",
            }
            for d in good_docs
        ] + [
            {
                "content": (
                    d.page_content[:500] + "..."
                    if len(d.page_content) > 500
                    else d.page_content
                ),
                "metadata": d.metadata,
                "origin": "web",
            }
            for d in web_docs
        ]

        return {
            "answer": state.get("answer", ""),
            "sources": sources,
            "crag_verdict": state.get("crag_verdict", ""),
            "crag_reason": state.get("crag_reason", ""),
            "issup": state.get("issup", ""),
            "evidence": state.get("evidence", []),
            "isuse": state.get("isuse", ""),
            "use_reason": state.get("use_reason", ""),
            "retries": state.get("retries", 0),
            "rewrite_tries": state.get("rewrite_tries", 0),
            
            # Feature 1 & 2 execution states
            "query_type": state.get("query_type", ""),
            "sql_query": state.get("sql_query", ""),
            "sql_query_id": state.get("sql_query_id", ""),
            "sql_status": state.get("sql_status", ""),
            "sql_results": state.get("sql_results", []),
            "approval_token": state.get("approval_token", ""),
            
            # Advanced Corrective RAG Config outputs
            "hyde_used": state.get("hyde_used", False),
            "hyde_hypotheses": state.get("hyde_hypotheses", []),
            "reranking_used": state.get("reranking_used", False)
        }
