import functools

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.core.graph.nodes import (
    router_node,
    sql_generation_node,
    ltm_remember_node,
    decide_retrieval_node,
    generate_direct_node,
    retrieve_docs_node,
    evaluate_docs_node,
    rewrite_query_node,
    web_search_node,
    refine_context_node,
    generate_answer_node,
    verify_support_node,
    revise_answer_node,
    verify_usefulness_node,
    rewrite_question_node,
    stm_summarize_node,
    hybrid_generation_node,
    route_after_router,
    route_after_decide,
    route_after_crag,
    route_after_support,
    route_after_usefulness,
)
from app.core.graph.state import CSRAGState
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph(
    vector_store: VectorStoreService,
    store: AsyncPostgresStore,
    checkpointer: AsyncPostgresSaver,
):
    builder = StateGraph(CSRAGState)

    retrieve_with_store = functools.partial(
        retrieve_docs_node, vector_store=vector_store
    )

    # 4-Path Routing Nodes
    builder.add_node("router", router_node)
    builder.add_node("sql_gen", sql_generation_node)
    builder.add_node("hybrid_gen", hybrid_generation_node)

    # CSRAG Nodes
    builder.add_node("ltm_remember", ltm_remember_node)
    builder.add_node("decide_retrieval", decide_retrieval_node)
    builder.add_node("generate_direct", generate_direct_node)
    builder.add_node("retrieve_docs", retrieve_with_store)
    builder.add_node("evaluate_docs", evaluate_docs_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("refine_context", refine_context_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("verify_support", verify_support_node)
    builder.add_node("revise_answer", revise_answer_node)
    builder.add_node("verify_usefulness", verify_usefulness_node)
    builder.add_node("rewrite_question", rewrite_question_node)
    builder.add_node("stm_summarize", stm_summarize_node)

    # Start Edge -> Route
    builder.add_edge(START, "router")

    # Dynamic classification route
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sql_gen": "sql_gen",
            "ltm_remember": "ltm_remember",
            "chat": "generate_direct",
            "hybrid": "hybrid_gen",
        },
    )

    # Terminate direct branch endpoints
    builder.add_edge("sql_gen", END)
    builder.add_edge(
        "generate_direct", "stm_summarize"
    )  # LOGIC-01: summarize conversation even for direct responses
    builder.add_edge("hybrid_gen", "stm_summarize")

    # Ingest document RAG pipeline sequence
    builder.add_edge("ltm_remember", "decide_retrieval")

    builder.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {
            "generate_direct": "generate_direct",
            "retrieve_docs": "retrieve_docs",
        },
    )

    builder.add_edge("retrieve_docs", "evaluate_docs")

    builder.add_conditional_edges(
        "evaluate_docs",
        route_after_crag,
        {
            "refine_context": "refine_context",
            "rewrite_query": "rewrite_query",
        },
    )

    builder.add_edge("rewrite_query", "web_search")
    builder.add_edge("web_search", "refine_context")
    builder.add_edge("refine_context", "generate_answer")
    builder.add_edge("generate_answer", "verify_support")

    builder.add_conditional_edges(
        "verify_support",
        route_after_support,
        {
            "revise_answer": "revise_answer",
            "verify_usefulness": "verify_usefulness",
        },
    )

    builder.add_edge("revise_answer", "verify_support")

    builder.add_conditional_edges(
        "verify_usefulness",
        route_after_usefulness,
        {
            "rewrite_question": "rewrite_question",
            "stm_summarize": "stm_summarize",
        },
    )

    builder.add_edge(
        "rewrite_question", "ltm_remember"
    )  # LOGIC-02: re-check LTM on rewritten query
    builder.add_edge("stm_summarize", END)

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    logger.info("IDOP 5-path LangGraph state machine compiled successfully")
    return graph
