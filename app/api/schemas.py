from datetime import datetime, timezone
from typing import Any, Literal, List, Optional, Dict
from pydantic import BaseModel, Field


# ==================== System Health & Diagnostics Schemas ====================

class HealthResponse(BaseModel):
    """System health check model detailing the service uptime status."""
    status: str = Field(
        ..., 
        description="Core operational status of the service (e.g. 'healthy', 'degraded', 'unhealthy')."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO 8601 UTC timestamp indicating exactly when the health check was performed."
    )
    version: str = Field(
        ..., 
        description="The semantic version string of the IDOP backend currently deployed."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2026-05-25T12:00:00Z",
                    "version": "0.1.0"
                }
            ]
        }
    }


class ReadinessResponse(BaseModel):
    """System readiness check model validating all database connections."""
    status: str = Field(
        ..., 
        description="Readiness state. 'ready' indicates all systems online; 'not_ready' implies connection drops."
    )
    qdrant_connected: bool = Field(
        ..., 
        description="Boolean connection status of the Qdrant hybrid vector store."
    )
    postgres_connected: bool = Field(
        ..., 
        description="Boolean connection status of the PostgreSQL database instance."
    )
    collection_info: dict = Field(
        ..., 
        description="Key-value dictionary showing the Qdrant collection status, vectors counts, and indexing state."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ready",
                    "qdrant_connected": True,
                    "postgres_connected": True,
                    "collection_info": {
                        "status": "green",
                        "vectors_count": 1420,
                        "indexed_chunks": 1420
                    }
                }
            ]
        }
    }


# ==================== Document Upload & Vector Collection Schemas ====================

class DocumentUploadResponse(BaseModel):
    """Response model for file ingestion detailing the vector indexing results."""
    message: str = Field(
        ..., 
        description="Human-readable execution message indicating successful ingestion or parse error."
    )
    filename: str = Field(
        ..., 
        description="The sanitised name of the file uploaded and parsed (e.g. 'Q2_Marketing_Guidelines.pdf')."
    )
    chunks_created: int = Field(
        ..., 
        description="The exact count of text chunks extracted, processed, and embedded."
    )
    document_ids: list[str] = Field(
        ..., 
        description="A list of unique UUID string hashes created for each chunk stored in Qdrant."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Successfully parsed and hybrid-indexed document.",
                    "filename": "Q2_Marketing_Guidelines.pdf",
                    "chunks_created": 42,
                    "document_ids": [
                        "8f8e8b0a-7f6c-5b4a-3a2b-1a0f9e8d7c6b",
                        "7f7e7b0b-6f5c-4b3a-2a1b-0a9f8e7d6c5b"
                    ]
                }
            ]
        }
    }


class CollectionInfoResponse(BaseModel):
    """Detailed vector collection model showing active dataset scale."""
    collection_name: str = Field(
        ..., 
        description="The active namespace name of the Qdrant collection (e.g. 'idop_documents')."
    )
    total_documents: int = Field(
        ..., 
        description="Total quantity of indexed text chunks stored across this collection."
    )
    status: str = Field(
        ..., 
        description="Status flag showing Qdrant performance state (e.g. 'green', 'yellow', 'red')."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "idop_documents",
                    "total_documents": 1420,
                    "status": "green"
                }
            ]
        }
    }


# ==================== Chat & RAG Interaction Schemas ====================

class ChatRequest(BaseModel):
    """The central request model for chatting, querying, and launching workflows."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="The user's question or command in natural language."
    )
    thread_id: str = Field(
        ..., 
        description=(
            "Conversation thread identifier. Supply the same thread ID across turns "
            "to preserve rolling Short-Term Memory. Generate a new UUID to clear history."
        )
    )
    user_id: str = Field(
        ..., 
        description=(
            "User identifier for Long-Term Memory. Facts and profile insights extracted "
            "from interactions are indexed against this user ID for personalization."
        )
    )
    include_sources: bool = Field(
        default=True, 
        description="Include references to source document chunks (Qdrant) or web items in response."
    )
    search_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid", 
        description="Search mode: dense (semantic), sparse (keyword), or hybrid (RRF fusion)"
    )
    top_k: int = Field(
        default=4, 
        ge=1, 
        le=20, 
        description="Quantity of documents to retrieve."
    )
    enable_hyde: bool = Field(
        default=False, 
        description="Use HYDE (Hypothetical Document Embeddings) for query expansion"
    )
    enable_reranking: bool = Field(
        default=False, 
        description="Use cross-encoder reranking for improved precision"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Show all SMB customers in Canada and check if their segment aligns with our sales strategy PDF.",
                    "thread_id": "thread-abc-123-uuid",
                    "user_id": "user-sales-manager-456",
                    "include_sources": True
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """Reference chunk or source link used to form the final answer."""
    content: str = Field(
        ..., 
        description="The raw text snippet retrieved from the document chunk (truncated to max 500 chars)."
    )
    metadata: dict[str, Any] = Field(
        ..., 
        description="Document metadata parameters containing filename, source path, score, or page numbers."
    )
    origin: Literal["internal", "web"] = Field(
        ..., 
        description="The source origin. 'internal' represents local Qdrant vectors; 'web' indicates Tavily search results."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "For SMB customer segment, the promotional discount must not exceed 15%...",
                    "metadata": {
                        "filename": "Marketing_Rules.pdf",
                        "page": 4,
                        "score": 0.89
                    },
                    "origin": "internal"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """The rich multi-channel response model returning synthesized answers, sources, and database metrics."""
    question: str = Field(
        ..., 
        description="The original natural language question requested by the user."
    )
    answer: str = Field(
        ..., 
        description="The beautifully formatted, synthesized markdown response compiled by the LLM."
    )
    sources: Optional[list[SourceDocument]] = Field(
        None, 
        description="The complete list of cited source document chunks or web links backing the answer."
    )
    processing_time_ms: float = Field(
        ..., 
        description="The end-to-end query processing and generation latency in milliseconds."
    )
    crag_verdict: str = Field(
        "", 
        description="Corrective RAG relevance classification verdict: CORRECT | AMBIGUOUS | INCORRECT."
    )
    crag_reason: str = Field(
        "", 
        description="Short justification of the CRAG relevance checker explaining vector score alignment."
    )
    issup: str = Field(
        "", 
        description="Self-Reflective RAG (SRAG) support verdict: fully_supported | partially_supported | no_support | skipped."
    )
    evidence: list[str] = Field(
        default_factory=list, 
        description="Direct quotes extracted from the text context proving the validity of the answer."
    )
    isuse: str = Field(
        "", 
        description="SRAG usefulness and user friendliness check verdict: useful | not_useful."
    )
    use_reason: str = Field(
        "", 
        description="Justification explaining whether the answer directly matches the user's intent."
    )
    retries: int = Field(
        0, 
        description="The quantity of self-reflection correction loops executed before finalizing output."
    )
    rewrite_tries: int = Field(
        0, 
        description="The quantity of query reformulations and Qdrant scroll iterations executed."
    )
    sql_query: Optional[str] = Field(
        None, 
        description="The SQL query statement compiled (returned only for hybrid RAG + SQL and pure SQL paths)."
    )
    sql_results: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="SQL execution result rows retrieved directly from PostgreSQL (only for hybrid)."
    )
    hyde_used: bool = Field(
        default=False, 
        description="True if HYDE query expansion was executed."
    )
    hyde_hypotheses: Optional[list[str]] = Field(
        None, 
        description="The hypothetical passages generated by the HyDE model."
    )
    reranking_used: bool = Field(
        default=False, 
        description="True if cross-encoder reranking was applied."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is our Q2 sales expectation?",
                    "answer": "According to the marketing guidelines, our Q2 sales goal is...",
                    "sources": [
                        {
                            "content": "Q2 sales guidelines mandate...",
                            "metadata": {"filename": "Q2_guidelines.pdf", "score": 0.85},
                            "origin": "internal"
                        }
                    ],
                    "processing_time_ms": 125.4,
                    "crag_verdict": "CORRECT",
                    "crag_reason": "Vector search returned exact matches in marketing docs.",
                    "issup": "fully_supported",
                    "evidence": ["Q2 sales guidelines mandate..."],
                    "isuse": "useful",
                    "use_reason": "Directly answers the user expectations.",
                    "retries": 0,
                    "rewrite_tries": 0,
                    "sql_query": None,
                    "sql_results": None
                }
            ]
        }
    }


class ChatMessage(BaseModel):
    """A single conversation turn storing conversation roles and message strings."""
    role: Literal["human", "assistant"] = Field(
        ..., 
        description="The sender role: 'human' for user input; 'assistant' for AI output."
    )
    content: str = Field(
        ..., 
        description="The text content of the message."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "human",
                    "content": "Hello! How does this platform work?"
                }
            ]
        }
    }


class ChatHistoryResponse(BaseModel):
    """The complete message history return payload for a specific thread."""
    thread_id: str = Field(
        ..., 
        description="The conversation thread identifier (UUID)."
    )
    messages: list[ChatMessage] = Field(
        ..., 
        description="The list of all conversation turns stored, sorted oldest to newest."
    )
    summary: str = Field(
        "", 
        description="The active rolling Short-Term Memory summary compressing older context turns."
    )
    message_count: int = Field(
        ..., 
        description="Total quantity of messages recorded in this conversation history."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thread_id": "thread-abc-123-uuid",
                    "messages": [
                        {"role": "human", "content": "Hi"},
                        {"role": "assistant", "content": "Hello! How can I assist you with your data operations today?"}
                    ],
                    "summary": "User greeted assistant, assistant offered help.",
                    "message_count": 2
                }
            ]
        }
    }


# ==================== Personalization & LTM Memory Schemas ====================

class MemoryItem(BaseModel):
    """A single persistent profile fact extracted and stored in Long-Term Memory."""
    data: str = Field(
        ..., 
        description="The compiled profile fact extracted from conversations (e.g. 'User prefers CSV files')."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": "User acts as a Sales Manager specializing in SMB segments."
                }
            ]
        }
    }


class MemoryListResponse(BaseModel):
    """The list of persistent personalization profile facts retrieved for a user."""
    user_id: str = Field(
        ..., 
        description="The user identifier associated with the memories."
    )
    memories: list[MemoryItem] = Field(
        ..., 
        description="The complete list of personalization profile facts extracted."
    )
    count: int = Field(
        ..., 
        description="Total count of profile facts recorded in Long-Term Memory."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user-sales-manager-456",
                    "memories": [
                        {"data": "User acts as a Sales Manager specializing in SMB segments."},
                        {"data": "User is based in Canada."}
                    ],
                    "count": 2
                }
            ]
        }
    }


class DeleteMemoryResponse(BaseModel):
    """Deletion notification showing successful memory clearance."""
    message: str = Field(
        ..., 
        description="Status message indicating memory successfully cleared."
    )
    user_id: str = Field(
        ..., 
        description="The user identifier whose memories were deleted."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Long-term memories successfully cleared for user.",
                    "user_id": "user-sales-manager-456"
                }
            ]
        }
    }


# ==================== IDOP SQL Endpoints (Feature 1) ====================

class SQLApprovalRequest(BaseModel):
    """Pydantic model for human-in-the-loop SQL execution approval."""
    query_id: str = Field(
        ..., 
        description="The unique session ID of the generated SQL query awaiting approval."
    )
    approved: bool = Field(
        ..., 
        description="Supply True to execute the query against Postgres; False will reject and cancel it."
    )
    token: str = Field(
        ..., 
        description="The cryptographic single-use session token generated by the approval gate."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "8f8e8b0a-7f6c-5b4a-3a2b-1a0f9e8d7c6b",
                    "approved": True,
                    "token": "cryptographic_validation_token_string"
                }
            ]
        }
    }


class SQLResponse(BaseModel):
    """SQL generation payload returning the compiled query and judge explanation."""
    query_id: str = Field(
        ..., 
        description="The unique session ID generated for this query session."
    )
    question: str = Field(
        ..., 
        description="The original user natural language question."
    )
    sql: str = Field(
        ..., 
        description="The generated SQL statement compiled by the Vanna agent."
    )
    explanation: str = Field(
        ..., 
        description="Detailed judge analysis, security comments, and SQL schema explanations."
    )
    status: str = Field(
        ..., 
        description="The workflow status state (e.g. 'pending_approval', 'error')."
    )
    cache_hit: bool = Field(
        default=False, 
        description="True if the SQL generation matched an existing query key in the cache."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "8f8e8b0a-7f6c-5b4a-3a2b-1a0f9e8d7c6b",
                    "question": "How many customers do we have?",
                    "sql": "SELECT COUNT(*) as customer_count FROM customers;",
                    "explanation": "Generates a standard SELECT statement counting all rows in table customers.",
                    "status": "pending_approval",
                    "cache_hit": False
                }
            ]
        }
    }


class SQLExecuteResponse(BaseModel):
    """The database results return payload after successful execution."""
    query_id: str = Field(
        ..., 
        description="The query session ID executed."
    )
    sql: str = Field(
        ..., 
        description="The SQL query statement executed."
    )
    results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="The complete list of query result rows retrieved from the database."
    )
    result_count: int = Field(
        ..., 
        description="The total row count of the retrieved dataset."
    )
    status: str = Field(
        ..., 
        description="Execution status flag (e.g. 'executed', 'failed', 'rejected')."
    )
    cache_hit: bool = Field(
        default=False, 
        description="True if the results were fetched directly from the SQL result cache."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "8f8e8b0a-7f6c-5b4a-3a2b-1a0f9e8d7c6b",
                    "sql": "SELECT COUNT(*) as customer_count FROM customers;",
                    "results": [
                        {"customer_count": 142}
                    ],
                    "result_count": 1,
                    "status": "executed",
                    "cache_hit": False
                }
            ]
        }
    }


# ==================== IDOP Mutation Endpoints (Feature 2) ====================

class MutationApprovalRequest(BaseModel):
    """Human-in-the-loop Excel spreadsheet mutation execution request."""
    mutation_id: str = Field(
        ..., 
        description="The unique session ID of the spreadsheet mutation pending approval."
    )
    approved: bool = Field(
        ..., 
        description="Supply True to execute mutations inside a safe transaction block; False will discard."
    )
    token: str = Field(
        ..., 
        description="The cryptographic single-use session token generated by the mutation approval gate."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mutation_id": "9f9e9b0a-8f7c-6b5a-4a3b-2a1f0e9d8c7b",
                    "approved": True,
                    "token": "mutation_validation_token_string"
                }
            ]
        }
    }


class MutationResponse(BaseModel):
    """Spradsheet parsing and business rules validation report."""
    mutation_id: str = Field(
        ..., 
        description="The unique session ID generated for this mutation task."
    )
    table_name: str = Field(
        ..., 
        description="The parsed destination database table targets (e.g. 'products')."
    )
    op_type: str = Field(
        ..., 
        description="The classified database operation type: INSERT | UPDATE | DELETE."
    )
    row_count: int = Field(
        ..., 
        description="The total quantity of rows parsed and alignment-checked from the file."
    )
    status: str = Field(
        ..., 
        description="Mutation workflow status (e.g. 'pending_approval', 'rules_violation')."
    )
    mappings: Dict[str, str] = Field(
        default_factory=dict, 
        description="Key-value mapping showing how file columns align with database fields."
    )
    errors: List[str] = Field(
        default_factory=list, 
        description="Detailed list of specific business rule violation messages (pricing boundaries, segments, etc.)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mutation_id": "9f9e9b0a-8f7c-6b5a-4a3b-2a1f0e9d8c7b",
                    "table_name": "products",
                    "op_type": "INSERT",
                    "row_count": 5,
                    "status": "pending_approval",
                    "mappings": {
                        "Product Name": "name",
                        "Category": "category",
                        "Price": "price",
                        "Stock": "stock_quantity"
                    },
                    "errors": []
                }
            ]
        }
    }


class MutationExecuteResponse(BaseModel):
    """Spreadsheet mutation transaction execution result."""
    mutation_id: str = Field(
        ..., 
        description="The mutation session ID executed."
    )
    rows_affected: int = Field(
        ..., 
        description="The quantity of rows successfully inserted, updated, or deleted."
    )
    status: str = Field(
        ..., 
        description="Commit state flag (e.g. 'executed', 'rolled_back', 'rejected')."
    )
    error: Optional[str] = Field(
        None, 
        description="Detailed Postgres execution exception message if database operations failed."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mutation_id": "9f9e9b0a-8f7c-6b5a-4a3b-2a1f0e9d8c7b",
                    "rows_affected": 5,
                    "status": "executed",
                    "error": None
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standardized operational error response payload."""
    error: str = Field(
        ..., 
        description="Operational error classification class (e.g. 'DatabaseError', 'ValidationError')."
    )
    message: str = Field(
        ..., 
        description="Human-readable summary of the failure."
    )
    detail: Optional[str] = Field(
        None, 
        description="Technical trace logs, exception codes, or parameters to aid debugging."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "DatabaseError",
                    "message": "Failed to connect to the relational database.",
                    "detail": "Connection timeout after 5000ms targeting postgres://..."
                }
            ]
        }
    }
