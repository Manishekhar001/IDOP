import json
import logging
from openai import OpenAI
from typing import Dict, List
from app.opik import track
from app.config import get_settings
from app.core.schema_registry import TABLE_SCHEMAS

logger = logging.getLogger("idop_app.column_mapper")


class ColumnMapper:
    """
    Performs LLM-based semantic mapping between messy spreadsheet headers and formal DB table schemas.
    """

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.memory_llm_model

    @track(name="column_mapper_map")
    def get_semantic_mapping(
        self, table_name: str, file_headers: List[str]
    ) -> Dict[str, str]:
        """
        Map spreadsheet headers to target DB table column names using semantic matching.
        """
        # Schema definitions from central registry — add new tables in schema_registry.py
        db_columns = TABLE_SCHEMAS.get(table_name, [])
        if not db_columns:
            raise ValueError(
                f"Target table '{table_name}' is not supported for mutations."
            )

        prompt = f"""
You are an expert data migration specialist.
Map the following user-supplied spreadsheet column headers to target database columns.

Target DB Table: {table_name}
Target DB Columns: {db_columns}

Spreadsheet Columns: {file_headers}

Establish matches using semantic meaning (e.g. "Unit Price (USD)" maps to "price", "Stock Qty" maps to "stock_quantity", "Segment" maps to "segment", etc.).
Do not match if there is no sensible mapping.

Respond strictly in this JSON format:
{{
  "mappings": {{
    "spreadsheet_column_name_1": "db_column_name_1",
    "spreadsheet_column_name_2": "db_column_name_2"
  }},
  "confidence": 0.0 to 1.0,
  "requires_review": true/false
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            mappings = result.get("mappings", {})
            logger.info(
                f"Generated semantic column mapping for {table_name}: {mappings}"
            )
            return mappings
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
