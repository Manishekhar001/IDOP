"""Seed a dedicated IDOP demo database and Qdrant collection with synthetic data.

Required environment:
  IDOP_DEMO_DATABASE_URL       A dedicated, empty PostgreSQL/Supabase demo database
  IDOP_DEMO_SEED_CONFIRM       Must equal I_AM_USING_A_DEDICATED_DEMO_DATABASE
  QDRANT_URL / QDRANT_API_KEY   Qdrant endpoint for the demo project
  Embedding-provider settings  NOMIC_API_KEY or VOYAGE_API_KEY

This script never drops tables or deletes a Qdrant collection. It refuses to
seed a business database that already contains rows unless its demo seed marker
is present. It always uses the isolated ``idop_demo_documents`` collection.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo_data"
DEMO_COLLECTION = "idop_demo_documents"
CONFIRMATION = "I_AM_USING_A_DEDICATED_DEMO_DATABASE"
DATASET_KEY = "idop-ecommerce-demo-v1"

BUSINESS_TABLES = ("customers", "products", "orders")
CSV_COLUMNS = {
    "customers": ("id", "name", "email", "segment", "country"),
    "products": ("id", "name", "category", "price", "stock_quantity", "description"),
    "orders": (
        "id",
        "customer_id",
        "order_date",
        "total_amount",
        "status",
        "shipping_address",
    ),
}


def _load_rows(table: str) -> list[dict[str, str]]:
    with (DEMO_DIR / f"{table}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_database(database_url: str) -> bool:
    """Create business tables and seed only a blank dedicated demo database.

    Returns True when rows were newly inserted and False when this dataset was
    already seeded.
    """
    # Import the project's canonical table definitions; this import has no DB
    # side effects. Infrastructure tables are created by IDOP at runtime.
    from scripts.init_db import CREATE_CUSTOMERS, CREATE_ORDERS, CREATE_PRODUCTS

    ddl = {
        "customers": CREATE_CUSTOMERS,
        "products": CREATE_PRODUCTS,
        "orders": CREATE_ORDERS,
    }
    connection = psycopg2.connect(database_url)
    try:
        with connection:
            with connection.cursor() as cursor:
                for table in BUSINESS_TABLES:
                    cursor.execute(ddl[table])
                cursor.execute("""CREATE TABLE IF NOT EXISTS idop_demo_seed_runs (
                           dataset_key VARCHAR(100) PRIMARY KEY,
                           created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                       )""")
                cursor.execute(
                    "SELECT 1 FROM idop_demo_seed_runs WHERE dataset_key = %s",
                    (DATASET_KEY,),
                )
                if cursor.fetchone():
                    print(
                        "Demo database already contains this dataset; skipping SQL seed."
                    )
                    return False

                counts = {}
                for table in BUSINESS_TABLES:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cursor.fetchone()[0]
                occupied = {table: count for table, count in counts.items() if count}
                if occupied:
                    raise RuntimeError(
                        "Refusing to seed because business tables are not empty: "
                        f"{occupied}. Point IDOP_DEMO_DATABASE_URL at a dedicated demo database."
                    )

                for table in BUSINESS_TABLES:
                    columns = CSV_COLUMNS[table]
                    placeholders = ", ".join(["%s"] * len(columns))
                    column_sql = ", ".join(columns)
                    insert_sql = (
                        f"INSERT INTO {table} ({column_sql}) "
                        f"VALUES ({placeholders})"
                    )
                    rows = _load_rows(table)
                    for row in rows:
                        cursor.execute(
                            insert_sql, tuple(row[column] for column in columns)
                        )
                    print(f"Seeded {len(rows)} synthetic {table} rows.")
                    cursor.execute(
                        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                        f"(SELECT MAX(id) FROM {table}), true)"
                    )

                cursor.execute(
                    "INSERT INTO idop_demo_seed_runs (dataset_key) VALUES (%s)",
                    (DATASET_KEY,),
                )
        return True
    finally:
        connection.close()


def seed_documents() -> int:
    """Index synthetic demo policies in the dedicated Qdrant collection."""
    from langchain_core.documents import Document

    from app.core.document_processor import DocumentProcessor
    from app.core.vector_store import VectorStoreService

    processor = DocumentProcessor()
    chunks: list[Document] = []
    policy_files = sorted((DEMO_DIR / "policies").glob("*.txt"))
    for path in policy_files:
        chunks.extend(processor.process_upload_bytes(path.read_bytes(), path.name))

    store = VectorStoreService(collection_name=DEMO_COLLECTION)
    ids = store.add_documents(chunks)
    print(
        f"Indexed {len(chunks)} synthetic policy chunks into Qdrant collection "
        f"'{DEMO_COLLECTION}' ({len(ids)} point IDs returned)."
    )
    return len(ids)


def main() -> None:
    # Load the root .env while retaining explicit demo-target checks.
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-only",
        action="store_true",
        help="Seed PostgreSQL only; skip policy document indexing in Qdrant.",
    )
    args = parser.parse_args()

    database_url = os.getenv("IDOP_DEMO_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("Set IDOP_DEMO_DATABASE_URL to a dedicated demo database.")
    if os.getenv("IDOP_DEMO_SEED_CONFIRM") != CONFIRMATION:
        raise SystemExit(
            "Set IDOP_DEMO_SEED_CONFIRM="
            + CONFIRMATION
            + " after verifying this is a dedicated demo database."
        )

    # Do not print the connection string; it may contain credentials.
    print("Seeding the explicitly configured demo database (connection hidden).")
    seed_database(database_url)
    if not args.database_only:
        seed_documents()


if __name__ == "__main__":
    main()
