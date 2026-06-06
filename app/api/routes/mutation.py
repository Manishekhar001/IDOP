import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.api.schemas import (
    MutationResponse,
    MutationApprovalRequest,
    MutationExecuteResponse,
    ErrorResponse,
)

# Feature 2 component imports
from app.core.feature2_mutation.op_classifier import OpClassifier
from app.core.feature2_mutation.file_parser import FileParser
from app.core.feature2_mutation.column_mapper import ColumnMapper
from app.core.feature2_mutation.rule_validator import RuleValidator
from app.core.feature2_mutation.mutation_generator import MutationGenerator
from app.core.feature2_mutation.llm_judge import MutationLLMJudge
from app.core.feature2_mutation.approval_gate import mutation_approval_gate as gate
from app.core.feature2_mutation.executor import MutationExecutor

from app.services.pending_store import pending_mutations as shared_pending_mutations
from app.opik import track
from app.utils.logger import get_logger

from app.core.schema_registry import SUPPORTED_MUTATION_TABLES

logger = get_logger(__name__)
router = APIRouter(prefix="/mutation", tags=["Database Mutations"])

# Shared services
parser = FileParser()
mapper = ColumnMapper()
validator = RuleValidator()
generator = MutationGenerator()
judge = MutationLLMJudge()
executor = MutationExecutor()


@router.post(
    "/upload",
    response_model=MutationResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid file, empty spreadsheet, or parameter error",
        },
        500: {"model": ErrorResponse, "description": "Processing or LLM error"},
    },
    summary="Upload Excel/CSV for mutation mapping, validation, and preview",
    description=(
        "Upload a spreadsheet containing mutation payload data. The pipeline: "
        "parse file → map columns to database schema → validate business rules → "
        "classify operation type (INSERT/UPDATE/DELETE) → LLM audit → generate SQL → "
        "return approval token. The mutation remains pending until POST /mutation/approve."
    ),
)
@track(name="upload_mutation")
async def upload_mutation(
    table_name: str = Form(
        ..., description="Target database table name (e.g. 'products', 'customers')"
    ),
    request_intent: str = Form(
        ...,
        description="Natural language description of the mutation intent (e.g. 'Add products', 'Update stock levels')",
    ),
    file: UploadFile = File(
        ...,
        description="Excel (.xlsx/.xls) or CSV spreadsheet containing the mutation payload data",
    ),
    max_bulk_rows: Optional[str] = Form(
        None,
        description="Maximum allowed rows to prevent resource exhaustion (default: 1000)",
    ),
    primary_key: Optional[str] = Form(
        "id", description="Primary key column name for UPDATE and DELETE operations"
    ),
    auto_map: Optional[str] = Form(
        "true",
        description="Enable automatic LLM-based column mapping from file headers to database columns",
    ),
    skip_validation: Optional[str] = Form(
        "false", description="Skip business rules validation checks"
    ),
) -> MutationResponse:
    """
    Upload and process a spreadsheet for database mutation (INSERT, UPDATE, or DELETE).

    Pipeline: parse file → column mapping → validate business rules → classify operation →
    LLM audit → generate SQL → store pending → return approval token

    Args:
        table_name: Target database table for the mutation.
        request_intent: Description of what the mutation should accomplish.
        file: Excel (.xlsx/.xls) or CSV file containing the payload data.
        max_bulk_rows: Maximum number of rows to process (default 1000).
        primary_key: Primary key column (default 'id').
        auto_map: If true, uses LLM to map file columns to database columns.
        skip_validation: If true, skips business rule validation.

    Returns:
        MutationResponse: Parsing results with column mappings, row count, operation type,
                         validation errors (if any), and approval token.

    Raises:
        HTTPException 400: If the file is empty, exceeds row limits, or parameter parsing fails.
        HTTPException 500: If file parsing, LLM, or internal processing fails.
    """
    # Safely parse max_bulk_rows
    parsed_max_bulk_rows = None
    if max_bulk_rows is not None:
        val = str(max_bulk_rows).strip()
        if val:
            try:
                parsed_max_bulk_rows = int(val)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="max_bulk_rows must be an integer."
                )

    # Safely parse auto_map
    parsed_auto_map = True
    if auto_map is not None:
        val = str(auto_map).strip().lower()
        if val == "false":
            parsed_auto_map = False
        elif val == "true" or val == "":
            parsed_auto_map = True
        else:
            try:
                parsed_auto_map = bool(int(val))
            except ValueError:
                parsed_auto_map = True

    # Safely parse skip_validation
    parsed_skip_validation = False
    if skip_validation is not None:
        val = str(skip_validation).strip().lower()
        if val == "true":
            parsed_skip_validation = True
        elif val == "false" or val == "":
            parsed_skip_validation = False
        else:
            try:
                parsed_skip_validation = bool(int(val))
            except ValueError:
                parsed_skip_validation = False

    logger.info(
        f"Mutation upload request. Table: {table_name}, Intent: {request_intent}, PK: {primary_key}, AutoMap: {parsed_auto_map}"
    )

    if table_name not in SUPPORTED_MUTATION_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Target table '{table_name}' is not supported for mutations. Supported tables: {list(SUPPORTED_MUTATION_TABLES)}",
        )

    try:
        content = await file.read()
        filename = file.filename

        # 1. Parse File rows and validate size limits
        rows = parser.parse_file(content, filename)
        if not rows:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        limit = parsed_max_bulk_rows or 1000
        if len(rows) > limit:
            raise HTTPException(
                status_code=400,
                detail=f"Spreadsheet contains {len(rows)} rows, which exceeds the allowed maximum of {limit} rows.",
            )

        # 2. Column Mapping (semantic LLM matching)
        headers = list(rows[0].keys())
        if parsed_auto_map:
            col_mappings = await mapper.get_semantic_mapping(table_name, headers)
        else:
            col_mappings = {h: h for h in headers}

        # Map spreadsheet rows to database keys
        mapped_rows = []
        for r in rows:
            mapped_row = {}
            for file_col, db_col in col_mappings.items():
                if file_col in r:
                    mapped_row[db_col] = r[file_col]
            mapped_rows.append(mapped_row)

        # 3. Business Rule Validation
        validation_errors = []
        if not parsed_skip_validation:
            is_valid, validation_errors = validator.validate_rows(
                table_name, mapped_rows
            )

        # 4. Classify Mutation Type (INSERT, UPDATE, DELETE)
        classifier = OpClassifier()
        op_type = await classifier.classify_operation(request_intent)

        # Generate SQL
        sql = ""
        params = []
        updates = []
        ids = []

        pk = primary_key or "id"
        if op_type == "INSERT":
            sql, params = generator.generate_insert(table_name, mapped_rows)
        elif op_type == "UPDATE":
            updates = generator.generate_update(table_name, mapped_rows, primary_key=pk)
        elif op_type == "DELETE":
            sql, ids = generator.generate_delete(
                table_name, mapped_rows, primary_key=pk
            )

        # 5. LLM Judge Audit Check (async)
        is_approved, explanation = await judge.audit_mutation(
            request_intent, table_name, op_type
        )
        if not is_approved:
            validation_errors.append(f"Audit Warning: {explanation}")

        # 6. Session setup (synchronous psycopg2 — offload to thread)
        mutation_id = str(uuid.uuid4())
        token = await asyncio.to_thread(gate.generate_session, mutation_id)

        shared_pending_mutations[mutation_id] = {
            "table_name": table_name,
            "op_type": op_type,
            "mapped_rows": mapped_rows,
            "sql": sql,
            "params": params,
            "updates": updates,
            "ids": ids,
            "token": token,
        }

        return MutationResponse(
            mutation_id=mutation_id,
            table_name=table_name,
            op_type=op_type,
            row_count=len(rows),
            status="pending_approval" if not validation_errors else "failed_validation",
            mappings=col_mappings,
            errors=validation_errors,
            token=token,
        )

    except ValueError as val_err:
        logger.warning(f"Validation failure in upload: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Mutation upload endpoint failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process mutation upload: {str(e)}"
        )


@router.post(
    "/approve",
    response_model=MutationExecuteResponse,
    responses={
        403: {
            "model": ErrorResponse,
            "description": "Invalid or expired approval token",
        },
        404: {"model": ErrorResponse, "description": "Mutation session not found"},
        500: {"model": ErrorResponse, "description": "Database transaction error"},
    },
    summary="Approve and execute database mutation inside atomic transaction",
    description=(
        "Approve or reject a pending spreadsheet mutation. Requires the cryptographic token "
        "returned by POST /mutation/upload. On approval, executes the mutation inside a "
        "safe all-or-nothing database transaction. On rejection, discards the pending mutation."
    ),
)
@track(name="approve_mutation")
async def approve_mutation(body: MutationApprovalRequest) -> MutationExecuteResponse:
    """
    Approve or reject a pending database mutation with cryptographic token verification.

    Pipeline: verify token → handle rejection → execute transaction → commit/rollback

    The mutation is executed inside a single database transaction. If any row fails,
    the entire transaction is rolled back, leaving the database in its original state.

    Args:
        body: The approval request containing mutation_id, approved flag, and cryptographic token.

    Returns:
        MutationExecuteResponse: Execution results with rows affected count, commit status,
                                 and optional error details on rollback.

    Raises:
        HTTPException 403: If the cryptographic token is invalid, expired, or already used.
        HTTPException 404: If the mutation session ID is not found in the pending register.
        HTTPException 500: If the database transaction fails (changes are rolled back).
    """
    logger.info(
        f"Mutation approval request for ID: {body.mutation_id}, Approved: {body.approved}"
    )

    # 1. Verify Cryptographic Token
    if body.approved:
        if not await asyncio.to_thread(
            gate.verify_and_close_session, body.mutation_id, body.token
        ):
            raise HTTPException(
                status_code=403,
                detail="Invalid, expired or already used cryptographic approval token.",
            )

    # 2. Handle Rejection
    if not body.approved:
        if body.mutation_id in shared_pending_mutations:
            del shared_pending_mutations[body.mutation_id]
        return MutationExecuteResponse(
            mutation_id=body.mutation_id, rows_affected=0, status="rejected"
        )

    # 3. Handle Transaction Execution
    if body.mutation_id not in shared_pending_mutations:
        raise HTTPException(
            status_code=404, detail="Mutation session not found in pending register."
        )

    session_info = shared_pending_mutations[body.mutation_id]
    table_name = session_info["table_name"]
    op_type = session_info["op_type"]

    try:
        rows_affected = 0
        if op_type == "INSERT":
            rows_affected = await asyncio.to_thread(
                executor.execute_insert_transaction,
                body.mutation_id,
                table_name,
                session_info["sql"],
                session_info["params"],
            )
        elif op_type == "UPDATE":
            rows_affected = await asyncio.to_thread(
                executor.execute_updates_transaction,
                body.mutation_id,
                table_name,
                session_info["updates"],
            )
        elif op_type == "DELETE":
            rows_affected = await asyncio.to_thread(
                executor.execute_delete_transaction,
                body.mutation_id,
                table_name,
                session_info["sql"],
                session_info["ids"],
            )

        # Remove from pending queue
        if body.mutation_id in shared_pending_mutations:
            del shared_pending_mutations[body.mutation_id]

        return MutationExecuteResponse(
            mutation_id=body.mutation_id, rows_affected=rows_affected, status="executed"
        )

    except Exception as e:
        logger.error(
            f"Mutation execution transaction failed - transaction rolled back successfully: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pending",
    summary="Get all pending database mutations",
    description="Returns all spreadsheet mutations awaiting human approval. Each entry includes the mutation ID, target table, and operation type for the approval workflow.",
)
@track(name="get_pending_mutations")
async def get_pending() -> list[dict]:
    """
    Retrieve all pending database mutations awaiting human approval.

    Returns a list of pending mutations with their session IDs, target tables,
    and classified operation types. Use POST /mutation/approve to approve or
    reject a specific mutation.

    Returns:
        list[dict]: List of pending mutation objects with mutation_id, table_name, and op_type fields.
    """
    return [
        {
            "mutation_id": mid,
            "table_name": info["table_name"],
            "op_type": info["op_type"],
        }
        for mid, info in shared_pending_mutations.items()
    ]
