import logging
from typing import List, Dict, Any, Tuple

from app.opik import track

logger = logging.getLogger("idop_app.mutation_generator")


class MutationGenerator:
    """
    Generates safe parameterised SQL statements for batch database mutations (INSERT/UPDATE/DELETE).
    """

    def __init__(self):
        pass

    @track(name="mutation_generator_insert")
    def generate_insert(
        self, table_name: str, mapped_rows: List[Dict[str, Any]]
    ) -> Tuple[str, List[Tuple[Any, ...]]]:
        """
        Generate parameterized bulk INSERT statement and corresponding flat parameter tuples.
        """
        if not mapped_rows:
            return "", []

        columns = list(mapped_rows[0].keys())
        col_list = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})"

        params = []
        for row in mapped_rows:
            params.append(tuple(row[c] for c in columns))

        logger.info(
            f"Generated parameterized SQL INSERT for '{table_name}' with {len(params)} value tuples."
        )
        return sql, params

    @track(name="mutation_generator_update")
    def generate_update(
        self,
        table_name: str,
        mapped_rows: List[Dict[str, Any]],
        primary_key: str = "id",
    ) -> List[Tuple[str, Tuple[Any, ...]]]:
        """
        Generate parameterized UPDATE statements for each row.
        Returns a list of tuples containing (sql_statement, parameters).
        """
        updates = []
        for idx, row in enumerate(mapped_rows):
            if primary_key not in row or row[primary_key] is None:
                raise ValueError(
                    f"Missing primary key '{primary_key}' in row index {idx} for UPDATE operation."
                )

            pk_val = row[primary_key]
            columns = [col for col in row.keys() if col != primary_key]

            set_clause = ", ".join([f"{col} = %s" for col in columns])
            sql = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = %s"

            params = [row[col] for col in columns]
            params.append(pk_val)

            updates.append((sql, tuple(params)))

        logger.info(
            f"Generated {len(updates)} parameterized UPDATE statements for table '{table_name}'."
        )
        return updates

    @track(name="mutation_generator_delete")
    def generate_delete(
        self,
        table_name: str,
        mapped_rows: List[Dict[str, Any]],
        primary_key: str = "id",
    ) -> Tuple[str, List[Any]]:
        """
        Generate parameterized DELETE statement for bulk keys.
        """
        ids_to_delete = []
        for idx, row in enumerate(mapped_rows):
            if primary_key not in row or row[primary_key] is None:
                raise ValueError(
                    f"Missing primary key '{primary_key}' in row index {idx} for DELETE operation."
                )
            ids_to_delete.append(row[primary_key])

        placeholders = ", ".join(["%s"] * len(ids_to_delete))
        sql = f"DELETE FROM {table_name} WHERE {primary_key} IN ({placeholders})"

        logger.info(
            f"Generated parameterized DELETE for '{table_name}' with {len(ids_to_delete)} keys."
        )
        return sql, ids_to_delete
