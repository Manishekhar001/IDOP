"""
IDOP Database Initialization Script.

Creates all business tables (customers, products, orders) and infrastructure
tables (idop_approval_tokens, idop_audit_logs) then inserts comprehensive
seed / dummy data so the Text2SQL feature has realistic data to query.

Usage:
    python scripts/init_db.py

Requires:
    - .env with SUPABASE_DB_URL / DATABASE_URL set
    - psycopg2 (included in requirements.txt)
"""

import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

setup_logging("INFO")
logger = get_logger("init_db")


# ======================================================================
# DDL: CREATE TABLE statements
# ======================================================================

CREATE_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    segment VARCHAR(50) NOT NULL CHECK (segment IN ('SMB', 'Enterprise', 'Individual')),
    country VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0.01),
    stock_quantity INTEGER NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    order_date DATE NOT NULL DEFAULT CURRENT_DATE,
    total_amount DECIMAL(10, 2) NOT NULL CHECK (total_amount >= 0),
    status VARCHAR(50) NOT NULL CHECK (status IN ('Pending', 'Delivered', 'Cancelled', 'Processing')),
    shipping_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_APPROVAL_TOKENS = """
CREATE TABLE IF NOT EXISTS idop_approval_tokens (
    query_id VARCHAR(100) PRIMARY KEY,
    token VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_AUDIT_LOGS = """
CREATE TABLE IF NOT EXISTS idop_audit_logs (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(100),
    question TEXT,
    sql_query TEXT,
    status VARCHAR(50),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_STATEMENTS = [
    ("customers", CREATE_CUSTOMERS),
    ("products", CREATE_PRODUCTS),
    ("orders", CREATE_ORDERS),
    ("idop_approval_tokens", CREATE_APPROVAL_TOKENS),
    ("idop_audit_logs", CREATE_AUDIT_LOGS),
]


# ======================================================================
# SEED DATA
# ======================================================================

SEED_CUSTOMERS = [
    # SMB customers
    ("MapleTech Solutions", "info@mapletech.ca", "SMB", "Canada"),
    ("Northern Lights Co", "hello@northernlights.com", "SMB", "Canada"),
    ("Bella Vita Bistro", "orders@bellavita.com", "SMB", "Italy"),
    ("Sunrise Bakery", "bake@sunrisebakery.com", "SMB", "USA"),
    ("GreenLeaf Landscaping", "team@greenleaf.com", "SMB", "Canada"),
    ("Prairie Goods Store", "contact@prairiegoods.com", "SMB", "USA"),
    ("Alpine Outfitters", "sales@alpineoutfitters.ch", "SMB", "Switzerland"),
    ("BlueWave Marketing", "info@bluewave.marketing", "SMB", "UK"),
    # Enterprise customers
    ("Acme Corp Global", "enterprise@acmecorp.com", "Enterprise", "USA"),
    ("MegaData Industries", "procurement@megadata.io", "Enterprise", "USA"),
    ("EuroTrans Logistics", "deals@eurotrans.eu", "Enterprise", "Germany"),
    ("NipponTech KK", "biz@nippontech.jp", "Enterprise", "Japan"),
    ("Apex Financial Group", "info@apexfin.com", "Enterprise", "Canada"),
    ("Pinnacle Health Inc", "supply@pinnaclehealth.com", "Enterprise", "USA"),
    # Individual customers
    ("Alice Johnson", "alice.j@email.com", "Individual", "Canada"),
    ("Bob Smith", "bob.smith@email.com", "Individual", "USA"),
    ("Clara Müller", "clara.m@email.de", "Individual", "Germany"),
    ("David Chen", "david.chen@email.com", "Individual", "Canada"),
    ("Elena Rossi", "elena.rossi@email.it", "Individual", "Italy"),
    ("Frank Williams", "frank.w@email.com", "Individual", "USA"),
    ("Grace Kim", "grace.kim@email.com", "Individual", "South Korea"),
    ("Hiroshi Tanaka", "hiroshi.t@email.jp", "Individual", "Japan"),
    ("Isabella Santos", "isabella.s@email.com", "Individual", "Brazil"),
    ("James Wilson", "james.w@email.com", "Individual", "UK"),
    ("Katherine Lee", "katherine.lee@email.com", "Individual", "Australia"),
    ("Liam O'Brien", "liam.ob@email.ie", "Individual", "Ireland"),
]

SEED_PRODUCTS = [
    # Electronics
    (
        "SmartPro Laptop",
        "Electronics",
        1299.99,
        45,
        "High-performance laptop with 16GB RAM and 512GB SSD",
    ),
    (
        "DataVault USB Drive 128GB",
        "Electronics",
        29.99,
        200,
        "Portable 128GB USB 3.0 flash drive",
    ),
    ('UltraView 27" Monitor', "Electronics", 449.99, 30, "27-inch 4K UHD IPS display"),
    (
        "Wireless Ergonomic Mouse",
        "Electronics",
        79.99,
        150,
        "Bluetooth ergonomic mouse with USB-C charging",
    ),
    (
        "Mechanical Keyboard Pro",
        "Electronics",
        149.99,
        80,
        "RGB backlit mechanical keyboard, Cherry MX switches",
    ),
    (
        "Noise-Canceling Headphones",
        "Electronics",
        299.99,
        60,
        "Over-ear Bluetooth headphones with ANC",
    ),
    ("WebCam 4K Pro", "Electronics", 89.99, 120, "4K webcam with built-in ring light"),
    # Software
    (
        "IDOP Enterprise Suite License",
        "Software",
        4999.99,
        10,
        "Annual license for IDOP data operations platform",
    ),
    (
        "DataSync Pro License",
        "Software",
        299.99,
        200,
        "One-year license for automated data synchronization",
    ),
    (
        "CyberShield Antivirus",
        "Software",
        49.99,
        500,
        "Enterprise-grade antivirus with real-time protection",
    ),
    (
        "CloudStorage 1TB Plan",
        "Software",
        119.99,
        300,
        "Annual 1TB cloud storage subscription",
    ),
    # Hardware
    (
        "Network Switch 48-Port",
        "Hardware",
        899.99,
        15,
        "Gigabit managed network switch, 48 ports",
    ),
    (
        "Server Rack 42U",
        "Hardware",
        1299.99,
        8,
        "Standard 42U server rack with cooling fans",
    ),
    (
        "CAT6 Ethernet Cable 10m",
        "Hardware",
        12.99,
        500,
        "High-speed CAT6 Ethernet cable, 10 meters",
    ),
    (
        "UPS Battery Backup 1500VA",
        "Hardware",
        359.99,
        25,
        "Uninterruptible power supply, 1500VA capacity",
    ),
    (
        "SSD 2TB NVMe",
        "Hardware",
        249.99,
        100,
        "NVMe M.2 solid-state drive, 2TB capacity",
    ),
    # Services
    (
        "Cloud Migration Service",
        "Services",
        9999.99,
        3,
        "Full cloud migration consulting and execution package",
    ),
    (
        "IT Support Monthly Retainer",
        "Services",
        2499.99,
        20,
        "Monthly premium IT support and maintenance retainer",
    ),
    (
        "Data Analytics Consulting",
        "Services",
        5000.00,
        5,
        "One-week data analytics strategy consulting engagement",
    ),
]

SEED_ORDERS = [
    # customer_id, order_date, total_amount, status, shipping_address
    # Delivered orders
    (1, "2026-01-15", 1299.99, "Delivered", "123 Maple Street, Toronto, ON, Canada"),
    (3, "2026-01-20", 4999.99, "Delivered", "Via Roma 42, Milan, Italy"),
    (5, "2026-02-01", 89.99, "Delivered", "456 Greenway Blvd, Vancouver, BC, Canada"),
    (9, "2026-02-10", 14999.98, "Delivered", "1 Acme Plaza, New York, NY, USA"),
    (10, "2026-02-15", 359.99, "Delivered", "500 Data Drive, San Francisco, CA, USA"),
    (2, "2026-02-20", 79.99, "Delivered", "100 Northern Ave, Edmonton, AB, Canada"),
    (4, "2026-03-01", 29.99, "Delivered", "789 Oak Street, Chicago, IL, USA"),
    (6, "2026-03-05", 449.99, "Delivered", "321 Prairie Lane, Dallas, TX, USA"),
    (15, "2026-03-10", 4999.99, "Delivered", "55 King Street W, Toronto, ON, Canada"),
    (16, "2026-03-15", 299.99, "Delivered", "100 Main St, Los Angeles, CA, USA"),
    (11, "2026-03-20", 1299.99, "Delivered", "Hauptstrasse 10, Berlin, Germany"),
    (19, "2026-03-25", 79.99, "Delivered", "Av Paulista 1000, São Paulo, Brazil"),
    (7, "2026-04-01", 149.99, "Delivered", "Bergstrasse 5, Zurich, Switzerland"),
    # Processing orders
    (12, "2026-04-05", 249.99, "Processing", "2-1 Marunouchi, Tokyo, Japan"),
    (20, "2026-04-08", 1299.99, "Processing", "221B Baker Street, London, UK"),
    (14, "2026-04-10", 449.99, "Processing", "88 Queen Street, Sydney, Australia"),
    # Pending orders
    (1, "2026-04-12", 899.99, "Pending", "123 Maple Street, Toronto, ON, Canada"),
    (8, "2026-04-14", 299.99, "Pending", "12 Thames Street, London, UK"),
    (17, "2026-04-15", 29.99, "Pending", "123 Gangnam-daero, Seoul, South Korea"),
    (13, "2026-04-16", 119.99, "Pending", "Via Roma 5, Rome, Italy"),
    # Cancelled orders
    (9, "2026-02-05", 5000.00, "Cancelled", "1 Acme Plaza, New York, NY, USA"),
    (18, "2026-03-30", 89.99, "Cancelled", "1-1 Chiyoda, Tokyo, Japan"),
]


# ======================================================================
# Main execution
# ======================================================================


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the public schema."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s)",
            (table_name,),
        )
        return cur.fetchone()[0]


def drop_all_tables(conn):
    """Drop all business tables in reverse dependency order."""
    logger.warning("Dropping all existing business tables...")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS idop_audit_logs CASCADE")
        cur.execute("DROP TABLE IF EXISTS idop_approval_tokens CASCADE")
        cur.execute("DROP TABLE IF EXISTS orders CASCADE")
        cur.execute("DROP TABLE IF EXISTS products CASCADE")
        cur.execute("DROP TABLE IF EXISTS customers CASCADE")
    conn.commit()
    logger.info("All business tables dropped.")


def create_all_tables(conn):
    """Create all business tables."""
    logger.info("Creating database tables...")
    for name, ddl in DDL_STATEMENTS:
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info(f"  ✓ Created table: {name}")
        except Exception as e:
            conn.rollback()
            logger.error(f"  ✗ Failed to create {name}: {e}")
            raise


def seed_table(conn, table_name: str, columns: list, rows: list):
    """Insert seed data into a table, skipping if data already exists."""
    # Count existing rows
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]

    if count > 0:
        logger.info(f"  - {table_name}: {count} rows already exist, skipping seed")
        return

    if not rows:
        logger.info(f"  - {table_name}: no seed data provided")
        return

    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

    try:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(insert_sql, row)
        conn.commit()
        logger.info(f"  ✓ {table_name}: seeded {len(rows)} rows")
    except Exception as e:
        conn.rollback()
        logger.error(f"  ✗ Failed to seed {table_name}: {e}")
        raise


def verify_tables(conn):
    """Verify all tables have data after seeding."""
    tables_to_check = [
        "customers",
        "products",
        "orders",
        "idop_approval_tokens",
        "idop_audit_logs",
    ]
    logger.info("\nVerifying database state...")
    all_ok = True
    for table in tables_to_check:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = %s",
                (table,),
            )
            exists = cur.fetchone()[0] > 0
            if exists:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                status = "✓" if count > 0 else "⚠"
                logger.info(f"  {status} {table}: {count} rows")
            else:
                logger.error(f"  ✗ {table}: TABLE DOES NOT EXIST")
                all_ok = False

    if all_ok:
        logger.info("\n✅ Database initialization complete!")
    else:
        logger.warning("\n⚠ Database initialization finished with warnings.")


def main():
    settings = get_settings()
    db_url = settings.supabase_db_url or settings.database_url

    if not db_url:
        logger.error("No DATABASE_URL or SUPABASE_DB_URL found in environment / .env")
        sys.exit(1)

    logger.info("Connecting to database...")
    conn = psycopg2.connect(db_url)

    try:
        # Read command-line action
        action = sys.argv[1].lower() if len(sys.argv) > 1 else "init"

        if action == "drop":
            drop_all_tables(conn)

        if action in ("init", "recreate"):
            if action == "recreate":
                drop_all_tables(conn)
            create_all_tables(conn)

            logger.info("\nSeeding data...")
            seed_table(
                conn,
                "customers",
                ["name", "email", "segment", "country"],
                SEED_CUSTOMERS,
            )
            seed_table(
                conn,
                "products",
                ["name", "category", "price", "stock_quantity", "description"],
                SEED_PRODUCTS,
            )
            seed_table(
                conn,
                "orders",
                [
                    "customer_id",
                    "order_date",
                    "total_amount",
                    "status",
                    "shipping_address",
                ],
                SEED_ORDERS,
            )
            # Infrastructure tables don't get seed data — they're filled at runtime

            verify_tables(conn)

        elif action == "seed":
            seed_table(
                conn,
                "customers",
                ["name", "email", "segment", "country"],
                SEED_CUSTOMERS,
            )
            seed_table(
                conn,
                "products",
                ["name", "category", "price", "stock_quantity", "description"],
                SEED_PRODUCTS,
            )
            seed_table(
                conn,
                "orders",
                [
                    "customer_id",
                    "order_date",
                    "total_amount",
                    "status",
                    "shipping_address",
                ],
                SEED_ORDERS,
            )
            verify_tables(conn)

        elif action == "verify":
            verify_tables(conn)

        else:
            print("Usage: python scripts/init_db.py [init|recreate|seed|verify|drop]")
            print("  init      - Create tables + seed data (default)")
            print("  recreate  - Drop + recreate tables + seed data")
            print("  seed      - Only insert seed data (skip table creation)")
            print("  verify    - Check table state")
            print("  drop      - Drop all business tables")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
