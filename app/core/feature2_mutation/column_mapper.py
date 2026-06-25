from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.llm_factory import get_memory_llm
from app.core.schema_registry import TABLE_SCHEMAS
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MappingResult(BaseModel):
    mappings: dict[str, str] = Field(
        ...,
        description="Mapping from spreadsheet column names to database column names.",
    )
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    requires_review: bool = Field(
        ..., description="Whether the mapping requires human review."
    )


_COLUMN_MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert data migration specialist.\n"
            "Map the following user-supplied spreadsheet column headers to target database columns.\n"
            "Establish matches using semantic meaning (e.g. 'Unit Price (USD)' maps to 'price', "
            "'Stock Qty' maps to 'stock_quantity').\n"
            "Do not match if there is no sensible mapping.",
        ),
        (
            "human",
            "Target DB Table: {table_name}\n"
            "Target DB Columns: {db_columns}\n"
            "\n"
            "Spreadsheet Columns: {file_headers}",
        ),
    ]
)


class ColumnMapper:
    """
    Performs LLM-based semantic mapping between messy spreadsheet headers and formal DB table schemas.
    """

    def __init__(self):
        self.llm = get_memory_llm()
        self._chain = _COLUMN_MAP_PROMPT | self.llm.with_structured_output(
            MappingResult
        )

    @track(name="column_mapper_map")
    async def get_semantic_mapping(
        self, table_name: str, file_headers: list[str]
    ) -> dict[str, str]:
        """
        Map spreadsheet headers to target DB table column names using semantic matching.
        """
        # Schema definitions from central registry — add new tables in schema_registry.py
        db_columns = TABLE_SCHEMAS.get(table_name, [])
        if not db_columns:
            raise ValueError(
                f"Target table '{table_name}' is not supported for mutations."
            )

        try:
            result: MappingResult = await self._chain.ainvoke(
                {
                    "table_name": table_name,
                    "db_columns": db_columns,
                    "file_headers": file_headers,
                }
            )
            logger.info(
                f"Generated semantic column mapping for {table_name}: {result.mappings}"
            )
            return result.mappings
        except Exception as e:
            logger.error(f"Failed to generate semantic column mapping: {e}")
            # Fallback to direct casing / basic strip matching
            fallback_mappings = {}
            for col in file_headers:
                normalized = col.strip().lower().replace(" ", "_")
                for db_col in db_columns:
                    if normalized == db_col or db_col in normalized:
                        fallback_mappings[col] = db_col
            return fallback_mappings
