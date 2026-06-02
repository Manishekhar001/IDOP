from typing import Annotated, Literal
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CSRAGState(TypedDict):
    # Core state
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    user_id: str
    ltm_context: str
    need_retrieval: bool
    question: str
    retrieval_query: str
    rewrite_tries: int
    docs: list[Document]
    good_docs: list[Document]
    crag_verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT", ""]
    crag_reason: str
    web_query: str
    web_docs: list[Document]
    strips: list[str]
    kept_strips: list[str]
    refined_context: str
    answer: str
    issup: Literal[
        "fully_supported", "partially_supported", "no_support", "skipped", ""
    ]
    evidence: list[str]
    retries: int
    # Advanced Corrective RAG Configs
    search_mode: Literal["dense", "sparse", "hybrid"]
    top_k: int
    enable_hyde: bool
    enable_reranking: bool
    enable_ragas: bool
    hyde_used: bool
    hyde_hypotheses: list[str]
    reranking_used: bool

    # Existing state continues...
    isuse: Literal["useful", "not_useful", ""]
    use_reason: str

    # IDOP 5-path Routing State
    query_type: Literal["SQL", "MUTATION", "RAG", "CHAT", "HYBRID", ""]

    # Feature 1 (NL-to-SQL) State
    sql_query: str
    sql_results: list[dict]
    sql_query_id: str
    sql_explanation: str
    sql_status: str  # pending_approval, executed, rejected, error

    # Feature 2 (Excel/CSV Mutations) State
    mutation_id: str
    mutation_table: str
    mutation_op: Literal["INSERT", "UPDATE", "DELETE", ""]
    mutation_rows: list[dict]
    mutation_mapped_rows: list[dict]
    mutation_status: str  # pending_approval, executed, rejected, error
    mutation_error: str
    mutation_result_count: int

    # Cryptographic Approval Token
    approval_token: str

    # SQL generation overrides
    explain: bool
    vanna_temperature: float
    vanna_seed: int
    vanna_top_p: float
