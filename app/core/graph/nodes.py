import asyncio
import json
import re
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.crag.evaluator import get_crag_evaluator
from app.core.crag.web_search import get_web_search_service
from app.core.graph.state import CSRAGState
from app.core.graph.router import QueryRouter
from app.core.memory.ltm import get_ltm_service
from app.core.memory.stm import get_stm_summarizer
from app.core.srag.verifier import get_srag_verifier
from app.core.vector_store import VectorStoreService

# Feature 1 & 2 Imports
from app.core.feature1_sql.vanna_service import TextToSQLService
from app.core.feature1_sql.sql_validator import SQLValidator
from app.core.feature1_sql.llm_judge import LLMJudge
from app.core.feature1_sql.approval_gate import approval_gate as gate
from app.core.feature1_sql.executor import SQLExecutor
from app.services.cache_init import get_query_cache

from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def _get_chat_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def _build_system_prompt(ltm_context: str, summary: str) -> str:
    base = (
        "You are a knowledgeable and helpful assistant with memory capabilities.\n\n"
        "Answer questions clearly and concisely using the provided context.\n"
        "If no context is available, use your general knowledge.\n"
        "If you don't know the answer, say so clearly.\n"
        "Do not make up information."
    )
    sections = []
    if ltm_context and ltm_context != "(empty)":
        sections.append(f"Long-term user memory:\n{ltm_context}")
    if summary:
        sections.append(f"Recent conversation summary:\n{summary}")
    if sections:
        return base + "\n\n" + "\n\n".join(sections)
    return base


# ---------------------------------------------------------------------------
# IDOP Top-Level Router Node
# ---------------------------------------------------------------------------


@track(name="graph_router")
async def router_node(state: CSRAGState) -> dict:
    """Classifies user input into SQL, MUTATION, RAG, CHAT, or HYBRID."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    question = last_human.content if last_human else ""
    router = QueryRouter()
    query_type = await asyncio.to_thread(router.route_query, question)
    return {"query_type": query_type, "question": question}


# ---------------------------------------------------------------------------
# Feature 1: SQL Generation Node
# ---------------------------------------------------------------------------


@track(name="graph_sql_generation")
async def sql_generation_node(state: CSRAGState) -> dict:
    """Generates SQL query, performs validation, runs LLM Judge, and creates approval session."""
    question = state["question"]
    logger.info(f"Feature 1 SQL Node triggered: '{question}'")

    from app.services.pending_store import pending_queries as shared_pending_queries

    validator = SQLValidator()
    judge = LLMJudge()

    try:
        # Generate raw SQL (uses direct OpenAI fallback when Vanna imports fail)
        sql_service = TextToSQLService(query_cache_service=get_query_cache())
        gen_res = await sql_service.generate_sql_for_approval(
            question=question,
            explain=state.get("explain", True),
            vanna_temperature=state.get("vanna_temperature", None),
            vanna_seed=state.get("vanna_seed", None),
            vanna_top_p=state.get("vanna_top_p", None),
        )
        sql = gen_res["sql"]
        query_id = gen_res["query_id"]

        # Validate query safety
        is_safe, error_msg = validator.validate(sql)
        if not is_safe:
            logger.warning(f"SQL safety violation: {error_msg}")
            return {
                "sql_query": sql,
                "sql_status": "error",
                "sql_explanation": error_msg,
            }

        # Run semantic audit judge
        is_correct, explanation = await asyncio.to_thread(
            judge.judge_sql, question, sql
        )
        if not is_correct:
            logger.warning(f"SQL semantic failure: {explanation}")
            # Still offer with warning or mark rejected
            explanation = f"⚠️ LLM Judge Warning: {explanation}"

        # Cryptographic Token Session
        token = gate.generate_session(query_id)

        # Store in SHARED pending_queries so /sql/approve route can find it
        shared_pending_queries[query_id] = {
            "question": question,
            "sql": sql,
            "status": "pending_approval",
            "token": token,
        }

        return {
            "sql_query": sql,
            "sql_query_id": query_id,
            "sql_status": "pending_approval",
            "sql_explanation": explanation,
            "approval_token": token,
        }

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {
            "sql_status": "error",
            "sql_explanation": f"Generation failed: {str(e)}",
        }


# ---------------------------------------------------------------------------
# Feature 2: Mutation Processing Node
# ---------------------------------------------------------------------------


@track(name="graph_mutation")
async def mutation_node(state: CSRAGState) -> dict:
    """
    (Legacy — no longer wired in the graph.)

    Bulk mutations require a file upload.  The LangGraph now routes MUTATION
    queries directly to the chat node, which instructs users to use the
    /mutation/upload API endpoint.
    """
    logger.info("Feature 2 Mutation Node triggered")

    return {
        "mutation_status": "requires_file_upload",
        "mutation_error": (
            "Bulk mutations require a CSV or Excel file upload.\n"
            "1. POST /mutation/upload with your file\n"
            "2. Review the preview (GET /mutation/pending)\n"
            "3. Approve via POST /mutation/approve"
        ),
    }


# ---------------------------------------------------------------------------
# LTM
# ---------------------------------------------------------------------------


@track(name="graph_ltm_remember")
async def ltm_remember_node(
    state: CSRAGState,
    config: RunnableConfig,
    *,
    store,
) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "default")
    ltm = get_ltm_service()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    user_message = last_human.content if last_human else ""

    await ltm.extract_and_store(store, user_id, user_message)
    ltm_context = await ltm.read_memories(store, user_id)

    logger.info(f"LTM remember done for user={user_id}")
    return {"user_id": user_id, "ltm_context": ltm_context}


# ---------------------------------------------------------------------------
# Retrieval decision
# ---------------------------------------------------------------------------


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True ONLY if the question requires specific private/domain documents.",
    )
    reason: str = Field(..., description="One-sentence justification.")


_DECIDE_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing classifier. Decide if the query requires retrieval from document store.",
        ),
        ("human", "Question: {question}"),
    ]
)


@track(name="graph_decide_retrieval")
async def decide_retrieval_node(state: CSRAGState) -> dict:
    question = state["question"]
    llm = _get_chat_llm()
    decider = _DECIDE_RETRIEVAL_PROMPT | llm.with_structured_output(RetrieveDecision)

    try:
        decision: RetrieveDecision = await decider.ainvoke({"question": question})
        need_retrieval = decision.should_retrieve
    except Exception as e:
        logger.error(f"decide_retrieval failed: {e} — defaulting to False")
        need_retrieval = False
    return {
        "question": question,
        "need_retrieval": need_retrieval,
        "retrieval_query": question,
    }


# ---------------------------------------------------------------------------
# Direct generation (no retrieval path)
# ---------------------------------------------------------------------------


@track(name="graph_generate_direct")
async def generate_direct_node(state: CSRAGState) -> dict:
    import uuid

    llm = _get_chat_llm()
    system_msg = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )
    messages = [SystemMessage(content=system_msg, id=str(uuid.uuid4()))] + list(
        state["messages"]
    )
    response = await llm.ainvoke(messages)
    answer = response.content
    return {
        "answer": answer,
        "issup": "skipped",
        "evidence": [],
    }


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------


@track(name="graph_retrieve_docs")
async def retrieve_docs_node(
    state: CSRAGState, *, vector_store: VectorStoreService
) -> dict:
    query = state.get("retrieval_query") or state["question"]

    # Get custom dynamic settings from state
    top_k = state.get("top_k") or get_settings().retrieval_k
    search_mode = state.get("search_mode") or "hybrid"
    enable_hyde = state.get("enable_hyde", False)
    enable_reranking = state.get("enable_reranking", False)

    logger.info(
        f"Retrieving docs for: '{query[:80]}' (top_k={top_k}, search_mode={search_mode}, hyde={enable_hyde}, reranking={enable_reranking})"
    )

    # Generate HyDE hypotheses if enabled!
    hyde_used = False
    hyde_hypotheses = []
    retrieval_query = query

    if enable_hyde:
        try:
            from app.core.feature3_rag.hyde import HydeService

            hyde_service = HydeService()
            # Generate hypothetical passages (async call)
            hyde_hypotheses = await hyde_service.generate_hypothetical_documents_async(
                query
            )
            if hyde_hypotheses:
                # Use the first hypothesis for query expansion
                retrieval_query = hyde_hypotheses[0]
                hyde_used = True
                logger.info(f"HyDE: Query expanded to '{retrieval_query[:80]}'")
        except Exception as e:
            logger.error(f"HyDE pipeline in node failed: {e}")

    # Call vector store search with k and search_mode!
    docs = await asyncio.to_thread(
        vector_store.search, retrieval_query, k=top_k, mode=search_mode
    )

    # Reranking if enabled!
    reranking_used = False
    if enable_reranking and docs:
        try:
            from app.core.feature3_rag.reranking import RerankingService

            reranker = RerankingService()
            docs = reranker.rerank(query, docs, top_k=min(top_k, len(docs)))
            reranking_used = True
            logger.info("Reranking completed successfully.")
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")

    # Perform Context Enrichment Window (pad with neighbors)
    try:
        from app.core.feature3_rag.context_enrichment import ContextEnrichmentService

        enricher = ContextEnrichmentService()
        docs = enricher.enrich_documents(
            docs, num_neighbors=1, chunk_overlap=get_settings().chunk_overlap
        )
    except Exception as e:
        logger.error(f"Context enrichment failed: {e}")

    logger.info(f"Retrieved and enriched {len(docs)} docs")
    return {
        "docs": docs,
        "hyde_used": hyde_used,
        "hyde_hypotheses": hyde_hypotheses,
        "reranking_used": reranking_used,
    }


# ---------------------------------------------------------------------------
# CRAG evaluation
# ---------------------------------------------------------------------------


@track(name="graph_evaluate_docs")
async def evaluate_docs_node(state: CSRAGState) -> dict:
    evaluator = get_crag_evaluator()
    verdict, reason, good_docs = await evaluator.evaluate(
        question=state["question"],
        docs=state.get("docs", []),
    )
    return {
        "crag_verdict": verdict,
        "crag_reason": reason,
        "good_docs": good_docs,
    }


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


@track(name="graph_rewrite_query")
async def rewrite_query_node(state: CSRAGState) -> dict:
    svc = get_web_search_service()
    web_query = await svc.rewrite_query(state["question"])
    return {"web_query": web_query}


@track(name="graph_web_search")
async def web_search_node(state: CSRAGState) -> dict:
    svc = get_web_search_service()
    query = state.get("web_query") or state["question"]
    web_docs = await svc.search(query)
    return {"web_docs": web_docs}


# ---------------------------------------------------------------------------
# Context refinement
# ---------------------------------------------------------------------------


def _decompose_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class BatchFilterResult(BaseModel):
    kept_indices: list[int] = Field(
        ...,
        description="0-based indices of sentences directly helping answer the question.",
    )


_BATCH_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter for a RAG system.\n"
            "Indices of sentences helping answer the question.\n"
            "Output JSON with kept_indices (list of integers).",
        ),
        (
            "human",
            "Question: {question}\n\nSentences:\n{sentences_json}",
        ),
    ]
)


@track(name="graph_refine_context")
async def refine_context_node(state: CSRAGState) -> dict:
    verdict = state.get("crag_verdict", "CORRECT")
    good_docs = state.get("good_docs", [])
    web_docs = state.get("web_docs", [])

    if verdict == "CORRECT":
        docs_to_use = good_docs
    elif verdict == "INCORRECT":
        docs_to_use = web_docs
    else:
        docs_to_use = good_docs + web_docs

    raw_context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    if not raw_context:
        return {"strips": [], "kept_strips": [], "refined_context": ""}

    strips = _decompose_to_sentences(raw_context)
    if not strips:
        return {"strips": [], "kept_strips": [], "refined_context": raw_context}

    llm = _get_chat_llm()
    filter_chain = _BATCH_FILTER_PROMPT | llm.with_structured_output(BatchFilterResult)

    kept = strips
    try:
        result: BatchFilterResult = await filter_chain.ainvoke(
            {
                "question": state["question"],
                "sentences_json": json.dumps(strips),
            }
        )
        valid_indices = {i for i in result.kept_indices if 0 <= i < len(strips)}
        kept = [strips[i] for i in sorted(valid_indices)]
    except Exception as e:
        logger.error(f"Batch sentence filter failed: {e} — keeping all")

    refined_context = "\n".join(kept)
    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}\n\n"
            "Answer the question using ONLY the provided context.\n"
            "If the context is empty, say: 'I don't have enough information.'",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@track(name="graph_generate_answer")
async def generate_answer_node(state: CSRAGState) -> dict:
    llm = _get_chat_llm()
    system_prompt = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )
    response = await (_RAG_PROMPT | llm).ainvoke(
        {
            "system_prompt": system_prompt,
            "context": state.get("refined_context", ""),
            "question": state["question"],
        }
    )
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# SRAG verification & usefulness
# ---------------------------------------------------------------------------


@track(name="graph_verify_support")
async def verify_support_node(state: CSRAGState) -> dict:
    verifier = get_srag_verifier()
    verdict, evidence = await verifier.verify_support(
        question=state["question"],
        context=state.get("refined_context", ""),
        answer=state["answer"],
    )
    return {"issup": verdict, "evidence": evidence}


@track(name="graph_revise_answer")
async def revise_answer_node(state: CSRAGState) -> dict:
    verifier = get_srag_verifier()
    revised = await verifier.revise_answer(
        question=state["question"],
        context=state.get("refined_context", ""),
        answer=state["answer"],
    )
    new_retries = state.get("retries", 0) + 1
    return {"answer": revised, "retries": new_retries}


@track(name="graph_verify_usefulness")
async def verify_usefulness_node(state: CSRAGState) -> dict:
    verifier = get_srag_verifier()
    verdict, reason = await verifier.verify_usefulness(
        question=state["question"],
        answer=state["answer"],
    )
    return {"isuse": verdict, "use_reason": reason}


# ---------------------------------------------------------------------------
# Question rewrite & STM summarization
# ---------------------------------------------------------------------------


class RewrittenQuestion(BaseModel):
    query: str = Field(..., description="Rewritten retrieval query.")


_REWRITE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The previous answer was not useful. Reformulate into a retrieval query.",
        ),
        ("human", "Original question: {question}"),
    ]
)


@track(name="graph_rewrite_question")
async def rewrite_question_node(state: CSRAGState) -> dict:
    llm = _get_chat_llm()
    chain = _REWRITE_QUESTION_PROMPT | llm.with_structured_output(RewrittenQuestion)
    try:
        result: RewrittenQuestion = await chain.ainvoke({"question": state["question"]})
        new_query = result.query
    except Exception as e:
        logger.error(f"rewrite_question failed: {e}")
        new_query = state["question"]

    new_tries = state.get("rewrite_tries", 0) + 1
    return {"retrieval_query": new_query, "rewrite_tries": new_tries}


@track(name="graph_stm_summarize")
async def stm_summarize_node(state: CSRAGState) -> dict:
    import uuid

    answer = state.get("answer", "")
    ai_msg = AIMessage(content=answer, id=str(uuid.uuid4()))

    summarizer = get_stm_summarizer()
    all_messages = list(state["messages"]) + [ai_msg]

    if summarizer.should_summarize(all_messages):
        new_summary, remove_ops = await summarizer.summarize(
            messages=all_messages,
            existing_summary=state.get("summary", ""),
        )
        return {"messages": [ai_msg] + remove_ops, "summary": new_summary}

    return {"messages": [ai_msg]}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Hybrid SQL + RAG Generation Node
# ---------------------------------------------------------------------------


@track(name="graph_hybrid_generation")
async def hybrid_generation_node(state: CSRAGState) -> dict:
    """
    Executes simultaneous Text-to-SQL database operations and RAG document search,
    then synthesizes both into a unified, source-cited comprehensive answer.
    """
    question = state["question"]
    logger.info(f"Hybrid SQL + RAG node triggered for question: '{question}'")

    sql_service = TextToSQLService(query_cache_service=get_query_cache())
    validator = SQLValidator()
    executor = SQLExecutor()

    # 1. SQL Generation & Execution
    sql_query = ""
    sql_results = []
    sql_status = "skipped"
    sql_error = None

    try:
        # Generate raw SQL
        logger.info("Hybrid: Generating SQL query...")
        # Generate raw SQL
        gen_res = await sql_service.generate_sql_for_approval(
            question=question,
            explain=state.get("explain", True),
            vanna_temperature=state.get("vanna_temperature", None),
            vanna_seed=state.get("vanna_seed", None),
            vanna_top_p=state.get("vanna_top_p", None),
        )
        sql = gen_res["sql"]
        query_id = gen_res["query_id"]

        # Validate query safety (SELECT only)
        is_safe, error_msg = validator.validate(sql)
        if is_safe:
            sql_query = sql
            # Since standard hybrid is read-only analytics, we run SELECT directly and log it
            if sql.strip().upper().startswith("SELECT"):
                logger.info(f"Hybrid: Safely executing read-only SQL: {sql}")
                sql_results = executor.execute_and_log(query_id, question, sql)
                sql_status = "executed"
            else:
                logger.warning(
                    "Hybrid: Generated query was not a SELECT statement. Skipped direct execution."
                )
                sql_status = "failed_safety"
                sql_error = "Only SELECT queries are supported in hybrid execution."
        else:
            logger.warning(f"Hybrid SQL safety violation: {error_msg}")
            sql_status = "failed_safety"
            sql_error = error_msg

    except Exception as e:
        logger.error(f"Hybrid SQL execution failed: {e}")
        sql_status = "error"
        sql_error = str(e)
    # 2. Document RAG Retrieval
    docs = []
    good_docs = []
    refined_context = ""
    crag_verdict = "INCORRECT"

    # Get custom dynamic settings from state
    top_k = state.get("top_k") or get_settings().retrieval_k
    search_mode = state.get("search_mode") or "hybrid"
    enable_hyde = state.get("enable_hyde", False)
    enable_reranking = state.get("enable_reranking", False)

    hyde_used = False
    hyde_hypotheses = []
    retrieval_query = question

    try:
        # Generate HyDE hypotheses if enabled!
        if enable_hyde:
            try:
                from app.core.feature3_rag.hyde import HydeService

                hyde_service = HydeService()
                hyde_hypotheses = (
                    await hyde_service.generate_hypothetical_documents_async(question)
                )
                if hyde_hypotheses:
                    retrieval_query = hyde_hypotheses[0]
                    hyde_used = True
                    logger.info(
                        f"Hybrid HyDE: Query expanded to '{retrieval_query[:80]}'"
                    )
            except Exception as hyde_err:
                logger.error(f"Hybrid HyDE failed: {hyde_err}")

        logger.info(
            f"Hybrid: Querying Qdrant vector store (top_k={top_k}, search_mode={search_mode})..."
        )
        settings = get_settings()
        vector_store = VectorStoreService()

        # Call vector store hybrid search (runs synchronously under asyncio.to_thread)
        docs = await asyncio.to_thread(
            vector_store.search, retrieval_query, k=top_k, mode=search_mode
        )

        # Reranking if enabled!
        reranking_used = False
        if enable_reranking and docs:
            try:
                from app.core.feature3_rag.reranking import RerankingService

                reranker = RerankingService()
                docs = reranker.rerank(question, docs, top_k=min(top_k, len(docs)))
                reranking_used = True
                logger.info("Hybrid: Reranking completed successfully.")
            except Exception as rerank_err:
                logger.error(f"Hybrid Reranking failed: {rerank_err}")

        # Context enrichment window
        try:
            from app.core.feature3_rag.context_enrichment import (
                ContextEnrichmentService,
            )

            enricher = ContextEnrichmentService()
            docs = enricher.enrich_documents(
                docs, num_neighbors=1, chunk_overlap=settings.chunk_overlap
            )
        except Exception as enrich_err:
            logger.error(f"Hybrid RAG context enrichment failed: {enrich_err}")

        # CRAG verification
        if docs:
            evaluator = get_crag_evaluator()
            verdict, reason, crag_good_docs = await evaluator.evaluate(question, docs)
            crag_verdict = verdict
            good_docs = crag_good_docs

            # Sentence refinement
            raw_context = "\n\n".join(d.page_content for d in good_docs).strip()
            if raw_context:
                strips = _decompose_to_sentences(raw_context)
                if strips:
                    llm = _get_chat_llm()
                    filter_chain = _BATCH_FILTER_PROMPT | llm.with_structured_output(
                        BatchFilterResult
                    )
                    try:
                        result = await filter_chain.ainvoke(
                            {"question": question, "sentences_json": json.dumps(strips)}
                        )
                        valid_indices = {
                            i for i in result.kept_indices if 0 <= i < len(strips)
                        }
                        kept = [strips[i] for i in sorted(valid_indices)]
                        refined_context = "\n".join(kept)
                    except Exception as filter_err:
                        logger.warning(
                            f"Sentence filter failed in hybrid: {filter_err}"
                        )
                        refined_context = raw_context
                else:
                    refined_context = raw_context
        else:
            logger.info("Hybrid: No document chunks retrieved.")

    except Exception as e:
        logger.error(f"Hybrid RAG retrieval failed: {e}")

    # 3. Final Synthesis Answering
    logger.info("Hybrid: Synthesizing database results and document chunks...")
    llm = _get_chat_llm()

    # Format database data for LLM
    db_context = "No database results retrieved."
    if sql_results:
        import pandas as pd

        df = pd.DataFrame(sql_results)
        db_context = f"SQL Query Executed:\n{sql_query}\n\nQuery Results (CSV Format):\n{df.to_csv(index=False)}"

    system_prompt = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )

    synthesis_prompt = f"""
You are an expert enterprise business analyst for IDOP.
Provide a unified, highly precise, and well-organized business report answering the user's question.
You MUST integrate the structured database results (SQL) and unstructured document guidelines (RAG) provided below.

Structured DB Context (PostgreSQL):
{db_context}
Status: {sql_status} {f'({sql_error})' if sql_error else ''}

Unstructured Doc Context (Qdrant RAG):
{refined_context if refined_context else 'No documentation context retrieved.'}

Answer the question thoroughly, citing specific numbers from the database results and guidelines from the documents.
Explain how the database facts align, match, or conflict with the manual guidelines.
Provide your answer in professional markdown with clear headings, bullet points, or tables.
"""

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=synthesis_prompt),
            ]
        )
        answer = response.content
    except Exception as e:
        logger.error(f"Hybrid synthesis prompt failed: {e}")
        answer = f"Failed to synthesize answer. SQL status: {sql_status}. Results count: {len(sql_results)}. RAG chunks: {len(docs)}."

    return {
        "sql_query": sql_query,
        "sql_results": sql_results,
        "sql_status": sql_status,
        "docs": docs,
        "good_docs": good_docs,
        "refined_context": refined_context,
        "crag_verdict": crag_verdict,
        "answer": answer,
        "hyde_used": hyde_used,
        "hyde_hypotheses": hyde_hypotheses,
        "reranking_used": reranking_used,
    }


# ---------------------------------------------------------------------------
# Routing Pure Logic Functions
# ---------------------------------------------------------------------------


def route_after_router(
    state: CSRAGState,
) -> Literal["sql_gen", "ltm_remember", "chat", "hybrid"]:
    q_type = state.get("query_type", "CHAT")
    if q_type == "SQL":
        return "sql_gen"
    elif q_type == "MUTATION":
        # Mutations require file uploads — route to chat so the LLM explains the workflow
        return "chat"
    elif q_type == "RAG":
        return "ltm_remember"
    elif q_type == "HYBRID":
        return "hybrid"
    else:
        return "chat"


def route_after_decide(
    state: CSRAGState,
) -> Literal["generate_direct", "retrieve_docs"]:
    return "retrieve_docs" if state["need_retrieval"] else "generate_direct"


def route_after_crag(state: CSRAGState) -> Literal["refine_context", "rewrite_query"]:
    return "refine_context" if state["crag_verdict"] == "CORRECT" else "rewrite_query"


def route_after_support(
    state: CSRAGState,
) -> Literal["revise_answer", "verify_usefulness"]:
    issup = state.get("issup", "fully_supported")
    retries = state.get("retries", 0)
    if issup != "fully_supported" and retries < get_settings().srag_max_retries:
        return "revise_answer"
    return "verify_usefulness"


def route_after_usefulness(
    state: CSRAGState,
) -> Literal["rewrite_question", "stm_summarize"]:
    isuse = state.get("isuse", "useful")
    rewrite_tries = state.get("rewrite_tries", 0)
    if isuse == "not_useful" and rewrite_tries < get_settings().max_rewrite_tries:
        return "rewrite_question"
    return "stm_summarize"
