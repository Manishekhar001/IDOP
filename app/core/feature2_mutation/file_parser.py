from io import BytesIO
from typing import Any

import pandas as pd

from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FileParser:
    """
    Parses user uploaded Excel or CSV data spreadsheets using pandas and openpyxl.
    """

    @track(name="file_parser_parse")
    def parse_file(self, file_content: bytes, filename: str) -> list[dict[str, Any]]:
        """
        Parse file bytes into a list of row dictionaries.

        Warning: loads entire file into memory. For large Excel files (>50MB)
        consider chunked processing via pd.read_excel(..., chunksize=...).
        """
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > 50:
            logger.warning(
                f"Large file ({file_size_mb:.1f} MB) loaded entirely into memory. "
                "Consider reducing file size for better performance."
            )
        logger.info(f"Parsing uploaded file: {filename} ({file_size_mb:.1f} MB)")
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_content))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(BytesIO(file_content))
        else:
            raise ValueError(
                "Unsupported file format. Please upload an Excel (.xlsx/.xls) or CSV (.csv) file."
            )

        # Clean NaN/Null values to standard Python None
        df = df.where(pd.notnull(df), None)
        rows = df.to_dict(orient="records")
        logger.info(f"Successfully parsed {len(rows)} rows from file")
        return rows
