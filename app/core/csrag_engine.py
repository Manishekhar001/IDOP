from langchain_core.messages import HumanMessage
import uuid
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.config import get_settings
from app.core.graph.builder import build_graph
from app.core.vector_store import VectorStoreService
from app.services.cache_init import get_query_cache
from app.utils.logger import get_logger
from app.core.feature3_rag.ragas_evaluator import get_ragas_evaluator
from app.opik import track, start_as_current_trace, start_as_current_span, opik_context

logger = get_logger(__name__)
settings = get_settings()

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
        logger.info("CSRAGEngine initialized with IDOP 5-path router")

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
        enable_ragas: bool = False,
        explain: bool = True,
        vanna_temperature: float = 0.0,
        vanna_seed: int = 42,
        vanna_top_p: float = 0.1,
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
            "enable_ragas": enable_ragas,
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
            "approval_token": "",
            # SQL generation overrides
            "explain": explain,
            "vanna_temperature": vanna_temperature,
            "vanna_seed": vanna_seed,
            "vanna_top_p": vanna_top_p,
        }

    @track(name="csrag_aquery")
    async def aquery(
        self,
        question: str,
        thread_id: str,
        user_id: str,
        search_mode: str = "hybrid",
        top_k: int = 4,
        enable_hyde: bool = False,
        enable_reranking: bool = False,
        enable_ragas: bool = False,
        explain: bool = True,
        vanna_temperature: float = 0.0,
        vanna_seed: int = 42,
        vanna_top_p: float = 0.1,
    ) -> dict:
        logger.info(
            f"async query — thread={thread_id}, user={user_id}, " f"q='{question[:80]}'"
        )

        query_cache = get_query_cache()
        cache_key = query_cache.get_rag_key(question, top_k) if query_cache else None
        if (
            query_cache
            and (query_cache.enabled or query_cache.use_local)
            and search_mode == "hybrid"
            and not enable_hyde
        ):
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
            enable_reranking=enable_reranking,
            enable_ragas=enable_ragas,
            explain=explain,
            vanna_temperature=vanna_temperature,
            vanna_seed=vanna_seed,
            vanna_top_p=vanna_top_p,
        )
        result = await self._graph.ainvoke(init_state, config)
        formatted = self._format_result(result)

        # RAGAS evaluation (if enabled)
        if enable_ragas:
            try:
                ragas_evaluator = get_ragas_evaluator()
                # Collect context texts from sources for faithfulness evaluation
                contexts = [s["content"] for s in formatted.get("sources", [])]
                ragas_result = await ragas_evaluator.evaluate(
                    question=question,
                    answer=formatted.get("answer", ""),
                    contexts=contexts,
                )
                if ragas_result:
                    formatted["ragas_scores"] = ragas_result.model_dump()
                    logger.info(
                        f"RAGAS evaluation: relevancy={ragas_result.answer_relevancy:.3f}, "
                        f"faithfulness={ragas_result.faithfulness:.3f}, "
                        f"precision={ragas_result.context_precision:.3f}"
                    )
                else:
                    formatted["ragas_scores"] = None
            except Exception as ragas_err:
                logger.error(f"RAGAS evaluation error: {ragas_err}")
                formatted["ragas_scores"] = None

        # Check post-verification gates before writing to Redis/local cache
        if (
            query_cache
            and (query_cache.enabled or query_cache.use_local)
            and formatted.get("query_type") == "RAG"
        ):
            crag_verdict = formatted.get("crag_verdict")
            issup = formatted.get("issup")
            isuse = formatted.get("isuse")

            if (
                crag_verdict == "CORRECT"
                and issup == "fully_supported"
                and isuse == "useful"
            ):
                query_cache.set(
                    cache_key, formatted, ttl=settings.cache_ttl_rag, cache_type="rag"
                )
                logger.info(
                    "✓ RAG Cache MISS - passed 3-tier quality gates and cached successfully."
                )
            else:
                logger.info(
                    f"✗ RAG Cache write skipped - failed quality gates (crag={crag_verdict}, sup={issup}, use={isuse})"
                )

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
        enable_ragas: bool = False,
        explain: bool = True,
        vanna_temperature: float = 0.0,
        vanna_seed: int = 42,
        vanna_top_p: float = 0.1,
    ):
        """
        Streaming version of aquery with manual Opik tracing.

        Manual tracing is used here instead of @track because the @track decorator
        cannot properly instrument async generators — it creates a span that closes
        immediately when the generator object is created, rather than capturing the
        actual streaming duration. By using start_as_current_trace/start_as_current_span
        directly inside the generator body, the span correctly covers the full
        lifecycle from first yield to generator exhaustion.
        """
        logger.info(
            f"streaming query — thread={thread_id}, user={user_id}, "
            f"q='{question[:80]}'"
        )

        # Build trace input metadata once
        trace_input = {
            "question": question[:200],
            "thread_id": thread_id,
            "user_id": user_id,
            "search_mode": search_mode,
            "top_k": top_k,
            "enable_hyde": enable_hyde,
            "enable_reranking": enable_reranking,
        }

        query_cache = get_query_cache()
        cache_key = query_cache.get_rag_key(question, top_k) if query_cache else None

        # ── Cache hit path (short span) ──────────────────────────────────
        if (
            query_cache
            and (query_cache.enabled or query_cache.use_local)
            and search_mode == "hybrid"
            and not enable_hyde
        ):
            cached_result = query_cache.get(cache_key, cache_type="rag")
            if cached_result:
                answer_text = cached_result.get("answer", "")
                if answer_text:
                    logger.info(f"RAG Cache HIT for stream: '{question[:50]}'")
                    with start_as_current_trace(
                        name="csrag_astream",
                        input=trace_input,
                        metadata={"cache_hit": True, "cost_saved": "$0.05"},
                    ):
                        with start_as_current_span(
                            name="csrag_astream_body",
                            input=trace_input,
                            metadata={"cache_hit": True},
                            tags=["stream", "cache_hit"],
                        ):
                            yield "⏺ cache_hit=true  cost_saved=$0.05\n"
                            yield answer_text

                            # Update span & trace with output info
                            char_count = len(answer_text)
                            opik_context.update_current_span(
                                output={
                                    "answer_length_chars": char_count,
                                    "cache_hit": True,
                                }
                            )
                            opik_context.update_current_trace(
                                output={
                                    "answer_length_chars": char_count,
                                    "cache_hit": True,
                                }
                            )
                    return

        # ── Cache miss path (full streaming span) ────────────────────────
        with start_as_current_trace(
            name="csrag_astream",
            input=trace_input,
            metadata={"cache_hit": False},
            thread_id=thread_id,
        ):
            with start_as_current_span(
                name="csrag_astream_body",
                input=trace_input,
                metadata={"cache_hit": False},
                tags=["stream", "cache_miss"],
            ):
                yield "⏺ cache_hit=false\n"

                config = self._build_config(thread_id, user_id)
                init_state = self._initial_state(
                    question=question,
                    search_mode=search_mode,
                    top_k=top_k,
                    enable_hyde=enable_hyde,
                    enable_reranking=enable_reranking,
                    enable_ragas=enable_ragas,
                    explain=explain,
                    vanna_temperature=vanna_temperature,
                    vanna_seed=vanna_seed,
                    vanna_top_p=vanna_top_p,
                )

                _ANSWER_NODES = {"generate_answer", "generate_direct"}
                char_count = 0
                had_error = False

                try:
                    async for msg, metadata in self._graph.astream(
                        init_state,
                        config,
                        stream_mode="messages",
                    ):
                        node = metadata.get("langgraph_node", "")
                        if (
                            node in _ANSWER_NODES
                            and hasattr(msg, "content")
                            and msg.content
                        ):
                            chunk = msg.content
                            char_count += len(chunk)
                            yield chunk
                except Exception as e:
                    had_error = True
                    logger.error(f"Streaming error: {e}", exc_info=True)
                    yield f"\n\n[Error: {type(e).__name__}: {str(e)}]"
                finally:
                    # Finalize span & trace with output metadata
                    span_output = {
                        "answer_length_chars": char_count,
                        "cache_hit": False,
                        "had_error": had_error,
                    }
                    trace_output = {
                        "answer_length_chars": char_count,
                        "cache_hit": False,
                        "had_error": had_error,
                    }
                    opik_context.update_current_span(output=span_output)
                    opik_context.update_current_trace(output=trace_output)

    async def run_with_state(self, state: dict, config: dict) -> dict:
        """
        Run the graph with a pre-built state dict.

        This is the public API for advanced use cases (e.g., ablation studies)
        that need to inject custom initial state (e.g., forcing retrieval)
        without accessing the private ``self._graph`` attribute.

        Args:
            state: A complete initial state dict (use _initial_state() as base).
            config: A LangGraph config dict with ``configurable.thread_id`` etc.

        Returns:
            The final state dict after the graph finishes execution.
        """
        return await self._graph.ainvoke(state, config)

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
            "sql_explanation": state.get("sql_explanation", ""),
            "sql_status": state.get("sql_status", ""),
            "sql_results": state.get("sql_results", []),
            "approval_token": state.get("approval_token", ""),
            # Mutation execution details
            "mutation_id": state.get("mutation_id", ""),
            "mutation_table": state.get("mutation_table", ""),
            "mutation_op": state.get("mutation_op", ""),
            "mutation_status": state.get("mutation_status", ""),
            "mutation_error": state.get("mutation_error", ""),
            "mutation_result_count": state.get("mutation_result_count", 0),
            # Context and Memories
            "ltm_context": state.get("ltm_context", ""),
            # Advanced Corrective RAG Config outputs
            "hyde_used": state.get("hyde_used", False),
            "hyde_hypotheses": state.get("hyde_hypotheses", []),
            "reranking_used": state.get("reranking_used", False),
            "ragas_scores": state.get("ragas_scores", None),
        }
