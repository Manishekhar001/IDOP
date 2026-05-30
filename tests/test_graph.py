"""
Unit tests for the IDOP LangGraph state machine.

Tests graph compilation, state shape, conditional routing functions,
and the CSRAGEngine format_result utility.
"""

from app.core.graph.state import CSRAGState
from app.core.graph.nodes import (
    route_after_decide,
    route_after_crag,
    route_after_support,
    route_after_usefulness,
)

# ═══════════════════════════════════════════════════════════════════════
# CSRAGState Shape Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCSRAGState:
    """Tests for the LangGraph TypedDict state definition."""

    def test_state_has_all_core_fields(self):
        """Test that CSRAGState declares all expected core RAG fields."""
        annotations = CSRAGState.__annotations__
        core_fields = [
            "messages",
            "summary",
            "user_id",
            "ltm_context",
            "need_retrieval",
            "question",
            "retrieval_query",
            "rewrite_tries",
            "docs",
            "good_docs",
        ]
        for field in core_fields:
            assert field in annotations, f"Missing core field: {field}"

    def test_state_has_crag_fields(self):
        """Test that CSRAGState declares CRAG verdict fields."""
        annotations = CSRAGState.__annotations__
        crag_fields = [
            "crag_verdict",
            "crag_reason",
            "web_query",
            "web_docs",
            "strips",
            "kept_strips",
            "refined_context",
        ]
        for field in crag_fields:
            assert field in annotations, f"Missing CRAG field: {field}"

    def test_state_has_srag_fields(self):
        """Test that CSRAGState declares SRAG verification fields."""
        annotations = CSRAGState.__annotations__
        srag_fields = [
            "answer",
            "issup",
            "evidence",
            "retries",
            "isuse",
            "use_reason",
        ]
        for field in srag_fields:
            assert field in annotations, f"Missing SRAG field: {field}"

    def test_state_has_advanced_rag_config_fields(self):
        """Test that CSRAGState declares advanced RAG configuration fields."""
        annotations = CSRAGState.__annotations__
        rag_config_fields = [
            "search_mode",
            "top_k",
            "enable_hyde",
            "enable_reranking",
            "hyde_used",
            "hyde_hypotheses",
            "reranking_used",
        ]
        for field in rag_config_fields:
            assert field in annotations, f"Missing RAG config field: {field}"

    def test_state_has_5path_routing_fields(self):
        """Test that CSRAGState declares query_type with HYBRID support."""
        annotations = CSRAGState.__annotations__
        assert "query_type" in annotations
        # Verify it's a Literal type that includes HYBRID
        type_str = str(annotations["query_type"])
        assert "HYBRID" in type_str, "query_type Literal must include HYBRID"

    def test_state_has_feature1_sql_fields(self):
        """Test that CSRAGState declares Feature 1 SQL fields."""
        annotations = CSRAGState.__annotations__
        sql_fields = [
            "sql_query",
            "sql_results",
            "sql_query_id",
            "sql_explanation",
            "sql_status",
        ]
        for field in sql_fields:
            assert field in annotations, f"Missing SQL field: {field}"

    def test_state_has_feature2_mutation_fields(self):
        """Test that CSRAGState declares Feature 2 mutation fields."""
        annotations = CSRAGState.__annotations__
        mutation_fields = [
            "mutation_id",
            "mutation_table",
            "mutation_op",
            "mutation_rows",
            "mutation_mapped_rows",
            "mutation_status",
            "mutation_error",
            "mutation_result_count",
        ]
        for field in mutation_fields:
            assert field in annotations, f"Missing mutation field: {field}"

    def test_state_has_approval_token(self):
        """Test that CSRAGState declares the cryptographic approval_token field."""
        annotations = CSRAGState.__annotations__
        assert "approval_token" in annotations


# ═══════════════════════════════════════════════════════════════════════
# Conditional Routing Function Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRouteAfterDecide:
    """Tests for the route_after_decide routing function."""

    def test_retrieval_needed_routes_to_retrieve_docs(self):
        state = {"need_retrieval": True}
        assert route_after_decide(state) == "retrieve_docs"

    def test_no_retrieval_routes_to_generate_direct(self):
        state = {"need_retrieval": False}
        assert route_after_decide(state) == "generate_direct"


class TestRouteAfterCrag:
    """Tests for the route_after_crag routing function."""

    def test_correct_verdict_routes_to_refine_context(self):
        state = {"crag_verdict": "CORRECT"}
        assert route_after_crag(state) == "refine_context"

    def test_ambiguous_verdict_routes_to_rewrite_query(self):
        state = {"crag_verdict": "AMBIGUOUS"}
        assert route_after_crag(state) == "rewrite_query"

    def test_incorrect_verdict_routes_to_rewrite_query(self):
        state = {"crag_verdict": "INCORRECT"}
        assert route_after_crag(state) == "rewrite_query"


class TestRouteAfterSupport:
    """Tests for the route_after_support routing function."""

    def test_fully_supported_routes_to_verify_usefulness(self):
        state = {"issup": "fully_supported", "retries": 0}
        assert route_after_support(state) == "verify_usefulness"

    def test_partially_supported_within_retries_routes_to_revise(self):
        state = {"issup": "partially_supported", "retries": 0}
        assert route_after_support(state) == "revise_answer"

    def test_no_support_within_retries_routes_to_revise(self):
        state = {"issup": "no_support", "retries": 1}
        assert route_after_support(state) == "revise_answer"

    def test_partially_supported_exceeding_retries_routes_to_verify_usefulness(self):
        state = {"issup": "partially_supported", "retries": 5}
        assert route_after_support(state) == "verify_usefulness"

    def test_missing_issup_defaults_to_verify_usefulness(self):
        state = {}
        assert route_after_support(state) == "verify_usefulness"


class TestRouteAfterUsefulness:
    """Tests for the route_after_usefulness routing function."""

    def test_useful_routes_to_stm_summarize(self):
        state = {"isuse": "useful", "rewrite_tries": 0}
        assert route_after_usefulness(state) == "stm_summarize"

    def test_not_useful_within_retries_routes_to_rewrite_question(self):
        state = {"isuse": "not_useful", "rewrite_tries": 0}
        assert route_after_usefulness(state) == "rewrite_question"

    def test_not_useful_exceeding_retries_routes_to_stm_summarize(self):
        state = {"isuse": "not_useful", "rewrite_tries": 5}
        assert route_after_usefulness(state) == "stm_summarize"

    def test_missing_isuse_defaults_to_stm_summarize(self):
        state = {}
        assert route_after_usefulness(state) == "stm_summarize"


# ═══════════════════════════════════════════════════════════════════════
# CSRAGEngine._format_result Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFormatResult:
    """Tests for the engine's static result formatter."""

    def test_format_result_returns_all_keys(self):
        from app.core.csrag_engine import CSRAGEngine

        state = {
            "answer": "IDOP is an enterprise platform.",
            "good_docs": [],
            "web_docs": [],
            "crag_verdict": "CORRECT",
            "crag_reason": "All chunks relevant.",
            "issup": "fully_supported",
            "evidence": ["Evidence line 1"],
            "isuse": "useful",
            "use_reason": "",
            "retries": 0,
            "rewrite_tries": 0,
            "query_type": "RAG",
            "sql_query": "",
            "sql_query_id": "",
            "sql_status": "",
            "sql_results": [],
            "approval_token": "",
            "hyde_used": False,
            "hyde_hypotheses": [],
            "reranking_used": False,
        }
        result = CSRAGEngine._format_result(state)

        assert result["answer"] == "IDOP is an enterprise platform."
        assert result["query_type"] == "RAG"
        assert result["crag_verdict"] == "CORRECT"
        assert result["issup"] == "fully_supported"
        assert result["isuse"] == "useful"
        assert result["sources"] == []
        assert result["hyde_used"] is False
        assert result["reranking_used"] is False

    def test_format_result_truncates_long_source_content(self):
        from app.core.csrag_engine import CSRAGEngine
        from langchain_core.documents import Document

        long_content = "A" * 1000
        state = {
            "answer": "Some answer",
            "good_docs": [Document(page_content=long_content, metadata={"page": 1})],
            "web_docs": [],
            "crag_verdict": "",
            "crag_reason": "",
            "issup": "",
            "evidence": [],
            "isuse": "",
            "use_reason": "",
            "retries": 0,
            "rewrite_tries": 0,
            "query_type": "RAG",
            "sql_query": "",
            "sql_query_id": "",
            "sql_status": "",
            "sql_results": [],
            "approval_token": "",
            "hyde_used": False,
            "hyde_hypotheses": [],
            "reranking_used": False,
        }
        result = CSRAGEngine._format_result(state)

        assert len(result["sources"]) == 1
        source_content = result["sources"][0]["content"]
        assert source_content.endswith("...")
        assert len(source_content) == 503  # 500 chars + "..."

    def test_format_result_separates_internal_and_web_sources(self):
        from app.core.csrag_engine import CSRAGEngine
        from langchain_core.documents import Document

        state = {
            "answer": "Combined answer",
            "good_docs": [
                Document(page_content="Internal doc", metadata={"file": "policy.pdf"})
            ],
            "web_docs": [
                Document(
                    page_content="Web result", metadata={"url": "https://example.com"}
                )
            ],
            "crag_verdict": "AMBIGUOUS",
            "crag_reason": "",
            "issup": "",
            "evidence": [],
            "isuse": "",
            "use_reason": "",
            "retries": 0,
            "rewrite_tries": 0,
            "query_type": "RAG",
            "sql_query": "",
            "sql_query_id": "",
            "sql_status": "",
            "sql_results": [],
            "approval_token": "",
            "hyde_used": True,
            "hyde_hypotheses": ["hyp1", "hyp2"],
            "reranking_used": True,
        }
        result = CSRAGEngine._format_result(state)

        assert len(result["sources"]) == 2
        assert result["sources"][0]["origin"] == "internal"
        assert result["sources"][1]["origin"] == "web"
        assert result["hyde_used"] is True
        assert result["reranking_used"] is True

    def test_format_result_handles_missing_optional_state(self):
        from app.core.csrag_engine import CSRAGEngine

        # Minimal state — many keys missing
        state = {"answer": "Minimal answer"}
        result = CSRAGEngine._format_result(state)

        assert result["answer"] == "Minimal answer"
        assert result["sources"] == []
        assert result["query_type"] == ""
        assert result["retries"] == 0


# ═══════════════════════════════════════════════════════════════════════
# CSRAGEngine._initial_state Tests
# ═══════════════════════════════════════════════════════════════════════


class TestInitialState:
    """Tests for the engine's initial state factory."""

    def test_default_initial_state(self):
        from app.core.csrag_engine import CSRAGEngine

        state = CSRAGEngine._initial_state("What is IDOP?")

        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "What is IDOP?"
        assert state["search_mode"] == "hybrid"
        assert state["top_k"] == 4
        assert state["enable_hyde"] is False
        assert state["enable_reranking"] is False
        assert state["query_type"] == ""
        assert state["sql_query"] == ""
        assert state["mutation_op"] == ""

    def test_custom_rag_config(self):
        from app.core.csrag_engine import CSRAGEngine

        state = CSRAGEngine._initial_state(
            "Find refund policy",
            search_mode="dense",
            top_k=8,
            enable_hyde=True,
            enable_reranking=True,
        )

        assert state["search_mode"] == "dense"
        assert state["top_k"] == 8
        assert state["enable_hyde"] is True
        assert state["enable_reranking"] is True

    def test_initial_state_all_counters_zeroed(self):
        from app.core.csrag_engine import CSRAGEngine

        state = CSRAGEngine._initial_state("Test query")

        assert state["rewrite_tries"] == 0
        assert state["retries"] == 0
        assert state["mutation_result_count"] == 0
        assert state["hyde_used"] is False
        assert state["reranking_used"] is False
