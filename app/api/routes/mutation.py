import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.api.schemas import MutationResponse, MutationApprovalRequest, MutationExecuteResponse, ErrorResponse

# Feature 2 component imports
from app.core.feature2_mutation.op_classifier import OpClassifier
from app.core.feature2_mutation.file_parser import FileParser
from app.core.feature2_mutation.column_mapper import ColumnMapper
from app.core.feature2_mutation.rule_validator import RuleValidator
from app.core.feature2_mutation.mutation_generator import MutationGenerator
from app.core.feature2_mutation.llm_judge import MutationLLMJudge
from app.core.feature2_mutation.approval_gate import MutationApprovalGate
from app.core.feature2_mutation.executor import MutationExecutor

from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/mutation", tags=["Database Mutations"])

# Shared services
parser = FileParser()
mapper = ColumnMapper()
validator = RuleValidator()
generator = MutationGenerator()
judge = MutationLLMJudge()
gate = MutationApprovalGate()
executor = MutationExecutor()

# Pending memory store for mutation sessions
pending_mutations: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/upload",
    response_model=MutationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload Excel/CSV for mutation mapping and validation",
)
async def upload_mutation(
    table_name: str = Form(..., description="Target database table name"),
    request_intent: str = Form(..., description="Description of mutation intent (e.g. Add products, Update stock)"),
    file: UploadFile = File(..., description="Excel/CSV spreadsheet containing payload data"),
) -> MutationResponse:
    logger.info(f"Mutation upload request. Table: {table_name}, Intent: {request_intent}")

    try:
        content = await file.read()
        filename = file.filename

        # 1. Parse File rows
        rows = parser.parse_file(content, filename)
        if not rows:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 2. Semantic Column Mapping
        headers = list(rows[0].keys())
        col_mappings = mapper.get_semantic_mapping(table_name, headers)

        # Map spreadsheet rows to database keys
        mapped_rows = []
        for r in rows:
            mapped_row = {}
            for file_col, db_col in col_mappings.items():
                if file_col in r:
                    mapped_row[db_col] = r[file_col]
            mapped_rows.append(mapped_row)

        # 3. Business Rule Validation
        is_valid, validation_errors = validator.validate_rows(table_name, mapped_rows)

        # 4. Classify Mutation Type (INSERT, UPDATE, DELETE)
        classifier = OpClassifier()
        op_type = classifier.classify_operation(request_intent)

        # Generate SQL
        sql = ""
        params = []
        updates = []
        ids = []

        if op_type == "INSERT":
            sql, params = generator.generate_insert(table_name, mapped_rows)
        elif op_type == "UPDATE":
            updates = generator.generate_update(table_name, mapped_rows, primary_key="id")
        elif op_type == "DELETE":
            sql, ids = generator.generate_delete(table_name, mapped_rows, primary_key="id")

        # 5. LLM Judge Audit Check
        is_approved, explanation = judge.audit_mutation(request_intent, table_name, op_type)
        if not is_approved:
            validation_errors.append(f"Audit Warning: {explanation}")

        # 6. Session setup
        mutation_id = str(uuid.uuid4())
        token = gate.generate_session(mutation_id)

        pending_mutations[mutation_id] = {
            "table_name": table_name,
            "op_type": op_type,
            "mapped_rows": mapped_rows,
            "sql": sql,
            "params": params,
            "updates": updates,
            "ids": ids,
            "token": token
        }

        return MutationResponse(
            mutation_id=mutation_id,
            table_name=table_name,
            op_type=op_type,
            row_count=len(rows),
            status="pending_approval" if not validation_errors else "failed_validation",
            mappings=col_mappings,
            errors=validation_errors,
            token=token
        )

    except ValueError as val_err:
        logger.warning(f"Validation failure in upload: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Mutation upload endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process mutation upload: {str(e)}")


@router.post(
    "/approve",
    response_model=MutationExecuteResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Approve and execute database mutation inside atomic transaction",
)
async def approve_mutation(body: MutationApprovalRequest) -> MutationExecuteResponse:
    logger.info(f"Mutation approval request for ID: {body.mutation_id}, Approved: {body.approved}")

    # 1. Verify Cryptographic Token
    if body.approved:
        if not gate.verify_and_close_session(body.mutation_id, body.token):
            raise HTTPException(status_code=403, detail="Invalid, expired or already used cryptographic approval token.")

    # 2. Handle Rejection
    if not body.approved:
        if body.mutation_id in pending_mutations:
            del pending_mutations[body.mutation_id]
        return MutationExecuteResponse(
            mutation_id=body.mutation_id,
            rows_affected=0,
            status="rejected"
        )

    # 3. Handle Transaction Execution
    if body.mutation_id not in pending_mutations:
        raise HTTPException(status_code=404, detail="Mutation session not found in pending register.")

    session_info = pending_mutations[body.mutation_id]
    table_name = session_info["table_name"]
    op_type = session_info["op_type"]

    try:
        rows_affected = 0
        if op_type == "INSERT":
            rows_affected = executor.execute_insert_transaction(
                body.mutation_id, table_name, session_info["sql"], session_info["params"]
            )
        elif op_type == "UPDATE":
            rows_affected = executor.execute_updates_transaction(
                body.mutation_id, table_name, session_info["updates"]
            )
        elif op_type == "DELETE":
            rows_affected = executor.execute_delete_transaction(
                body.mutation_id, table_name, session_info["sql"], session_info["ids"]
            )

        # Remove from pending queue
        if body.mutation_id in pending_mutations:
            del pending_mutations[body.mutation_id]

        return MutationExecuteResponse(
            mutation_id=body.mutation_id,
            rows_affected=rows_affected,
            status="executed"
        )

    except Exception as e:
        logger.error(f"Mutation execution transaction failed - transaction rolled back successfully: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pending",
    summary="Get all pending database mutations",
)
async def get_pending():
    return [{"mutation_id": mid, "table_name": info["table_name"], "op_type": info["op_type"]} for mid, info in pending_mutations.items()]
