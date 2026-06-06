import logging
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd

# Vanna imports are lazy-loaded inside VannaAgentWrapper to avoid ImportError
# when vanna 2.0 submodules are not available.

from app.config import get_settings
from app.opik import track

logger = logging.getLogger("idop_app.vanna_service")


class VannaAgentWrapper:
    """
    Wrapper around Vanna 2.0 Agent for Text-to-SQL generation.

    All Vanna imports are lazy-loaded to gracefully handle missing/misconfigured
    vanna submodules. When Vanna is unavailable, callers fall back to direct
    OpenAI SQL generation (handled in TextToSQLService.generate_sql_for_approval).
    """

    def __init__(
        self,
        openai_api_key: str,
        database_url: str,
    ):
        self._available = False
        self.agent = None
        self.postgres_runner = None
        self.current_temperature = 0.0
        self.current_top_p = 0.1
        self.current_seed = 42

        try:
            from vanna import Agent as VannaAgent
            from vanna.integrations.openai import OpenAILlmService
            from vanna.integrations.postgres import PostgresRunner
            from vanna.core.registry import ToolRegistry
            from vanna.tools import RunSqlTool
            from vanna.integrations.local.agent_memory import DemoAgentMemory
            from vanna.core.user.resolver import UserResolver
            from vanna.core.user.models import User

            settings = get_settings()
            self.llm = OpenAILlmService(
                api_key=openai_api_key, model=settings.llm_model
            )

            logger.info(
                f"Configuring SQL LLM with deterministic settings: "
                f"temperature={settings.llm_temperature}"
            )

            self.current_temperature = settings.llm_temperature
            self.current_top_p = 0.1
            self.current_seed = 42

            original_build_payload = self.llm._build_payload

            def deterministic_build_payload(request):
                payload = original_build_payload(request)
                payload["temperature"] = self.current_temperature
                payload["top_p"] = self.current_top_p
                payload["seed"] = self.current_seed
                payload["max_tokens"] = 2000
                logger.debug(f"SQL LLM payload: {payload}")
                return payload

            self.llm._build_payload = deterministic_build_payload
            self.postgres_runner = PostgresRunner(connection_string=database_url)

            self.tools = ToolRegistry()
            self.tools.register_local_tool(
                RunSqlTool(sql_runner=self.postgres_runner),
                access_groups=["admin", "user"],
            )

            # Create a simple user resolver (Vanna 2.x requires this)
            class SimpleUserResolver(UserResolver):
                """Minimal UserResolver that returns a default user."""

                def resolve_user(self, request_context) -> User:
                    return User(
                        id="default",
                        data={"user_id": "default", "name": "IDOP User"},
                    )

            self.agent = VannaAgent(
                llm_service=self.llm,
                tool_registry=self.tools,
                agent_memory=DemoAgentMemory(),
                user_resolver=SimpleUserResolver(),
            )
            self._available = True
            logger.info("Vanna Agent Wrapper initialized successfully")
        except ImportError as e:
            logger.warning(
                f"Vanna 2.0 submodules not available ({e}) — will use direct LLM fallback"
            )
        except Exception as e:
            logger.warning(
                f"Vanna Agent initialization failed ({e}) — will use direct LLM fallback"
            )

    @track(name="vanna_generate_sql")
    async def generate_sql_async(self, question: str, schema_context: str = "") -> str:
        if not self._available or self.agent is None:
            raise RuntimeError("Vanna agent not available — use direct LLM fallback")

        if schema_context:
            full_message = f"{schema_context}\n\n Question: {question}"
        else:
            full_message = question

        try:
            from vanna.core.user import RequestContext
        except ImportError:
            raise RuntimeError("Vanna RequestContext not available")

        request_context = RequestContext()
        sql = None

        async for component in self.agent.send_message(
            request_context=request_context, message=full_message
        ):
            rich_comp = component.rich_component
            if hasattr(rich_comp, "metadata") and rich_comp.metadata:
                if "sql" in rich_comp.metadata:
                    sql = rich_comp.metadata["sql"]

            if hasattr(rich_comp, "content") and rich_comp.content:
                content = str(rich_comp.content)
                if "```sql" in content.lower():
                    parts = content.split("```")
                    for part in parts:
                        if part.strip().lower().startswith("sql"):
                            sql = part[3:].strip()

        if not sql:
            raise ValueError(
                "Agent did not generate SQL. Please try rephrasing your question"
            )
        return sql

    @track(name="vanna_execute_sql")
    async def execute_sql_async(self, sql: str) -> List[Dict[str, Any]]:
        if not self._available or self.postgres_runner is None:
            raise RuntimeError(
                "Vanna agent not available — cannot execute SQL through Vanna"
            )

        logger.info(f"Executing SQL directly: {sql[:100]}...")
        try:
            import psycopg2
            import psycopg2.extras
            import socket
            from urllib.parse import urlparse

            conn_str = self.postgres_runner.connection_string
            parsed = urlparse(conn_str)
            hostname = parsed.hostname

            try:
                logger.debug(f"Resolving hostname {hostname} to IPv4...")
                addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET)
                ipv4_address = addr_info[0][4][0]
                logger.info(f"Resolved {hostname} to IPv4: {ipv4_address}")
                conn_str = conn_str.replace(hostname, ipv4_address)
            except Exception as e:
                logger.warning(
                    f"Failed to resolve hostname to IPv4: {e}, using original hostname"
                )

            conn = psycopg2.connect(conn_str)
            try:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(sql)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                cursor.close()
                conn.close()
                logger.info(f"SQL executed successfully: {len(results)} rows returned")
                return results
            except Exception as e:
                conn.close()
                raise e
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise ValueError(f"Failed to execute SQL: {str(e)}")


class TextToSQLService:
    def __init__(
        self,
        database_url: str | None = None,
        openai_api_key: str | None = None,
        query_cache_service=None,
    ):
        settings = get_settings()
        self.database_url = database_url or settings.supabase_db_url
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.query_cache_service = query_cache_service

        if not self.database_url:
            raise ValueError("DATABASE_URL is required for Text-to-SQL features.")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for Text-to-SQL features.")

        self.vanna = VannaAgentWrapper(
            openai_api_key=self.openai_api_key,
            database_url=self.database_url,
        )
        self.is_trained = False
        self.schema_context = ""

    def complete_training(self):
        logger.info("Preparing schema context for Vanna 2.0...")
        self.schema_context = self._build_schema_context()
        self.is_trained = True
        logger.info("Schema prepared for Vanna 2.0 Agent!")

    def _build_schema_context(self) -> str:
        """
        Build schema context dynamically by introspecting the actual database.

        Queries information_schema.columns and information_schema.table_constraints
        to build an accurate, up-to-date representation of all business tables.
        Falls back to the hardcoded static context if the database is unreachable.
        """
        import psycopg2
        import psycopg2.extras

        schema_parts = []
        schema_parts.append("DATABASE SCHEMA DOCUMENTATION")
        schema_parts.append("=" * 60)

        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Fetch all business tables (exclude LangGraph / idop_ infrastructure tables)
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name NOT LIKE 'checkpoint_%'
                  AND table_name NOT LIKE 'store%'
                  AND table_name NOT LIKE 'idop_%'
                ORDER BY table_name
            """)
            business_tables = [r["table_name"] for r in cur.fetchall()]

            if not business_tables:
                logger.warning(
                    "No business tables found in database — using static schema fallback"
                )
                conn.close()
                return self._static_schema_context()

            # Build table relationships description
            schema_parts.append(
                "\nThis is an e-commerce database with the following tables:"
            )
            for tbl in business_tables:
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s",
                    (tbl,),
                )
                col_count = cur.fetchone()["cnt"]
                schema_parts.append(f"- {tbl}: {col_count} columns")

            # Fetch foreign keys
            cur.execute("""
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                  AND tc.table_name NOT LIKE 'checkpoint_%'
                  AND tc.table_name NOT LIKE 'store%'
                  AND tc.table_name NOT LIKE 'idop_%'
            """)
            foreign_keys = cur.fetchall()

            # Fetch column details for each table
            for tbl in business_tables:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (tbl,),
                )
                columns = cur.fetchall()

                schema_parts.append(f"\nTable: {tbl}")
                schema_parts.append("Columns:")
                for col in columns:
                    nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
                    default = (
                        f" DEFAULT {col['column_default']}"
                        if col["column_default"]
                        else ""
                    )
                    schema_parts.append(
                        f"  - {col['column_name']} ({col['data_type']}, {nullable}{default})"
                    )

                # Link foreign keys for this table
                for fk in foreign_keys:
                    if fk["table_name"] == tbl:
                        schema_parts.append(
                            f"  * Foreign Key: {fk['column_name']} -> {fk['foreign_table_name']}.{fk['foreign_column_name']}"
                        )

            # Fetch enum-like check constraint values
            cur.execute("""
                SELECT
                    tc.table_name,
                    ccu.column_name,
                    pg_get_constraintdef(tc.oid) as constraint_def
                FROM pg_catalog.pg_constraint tc
                JOIN information_schema.table_constraints tc2
                    ON tc.conname = tc2.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON tc2.constraint_name = ccu.constraint_name
                WHERE tc2.constraint_type = 'CHECK'
                  AND tc2.table_schema = 'public'
                  AND tc2.table_name NOT LIKE 'checkpoint_%'
                  AND tc2.table_name NOT LIKE 'store%'
                  AND tc2.table_name NOT LIKE 'idop_%'
                  AND ccu.column_name IN ('segment', 'status')
            """)
            check_constraints = cur.fetchall()

            if check_constraints:
                schema_parts.append("\nColumn Constraints (allowed values):")
                enum_descriptions = {
                    "customers.segment": "'SMB', 'Enterprise', 'Individual'",
                    "orders.status": "'Pending', 'Delivered', 'Cancelled', 'Processing'",
                }
                for cc in check_constraints:
                    key = f"{cc['table_name']}.{cc['column_name']}"
                    desc = enum_descriptions.get(
                        key, f"See check constraint: {cc['constraint_def']}"
                    )
                    schema_parts.append(f"  - {key}: {desc}")

            conn.close()

        except Exception as e:
            logger.warning(
                f"Dynamic schema introspection failed ({e}) — using static fallback"
            )
            return self._static_schema_context()

        # Add example queries
        schema_parts.append("\nEXAMPLE QUERIES:")
        schema_parts.append("-" * 60)

        examples = [
            (
                "How many customers do we have?",
                "SELECT COUNT(*) as customer_count FROM customers;",
            ),
            (
                "What is the total revenue from all orders?",
                "SELECT SUM(total_amount) as total_revenue FROM orders;",
            ),
            (
                "List all delivered orders",
                "SELECT * FROM orders WHERE status = 'Delivered' ORDER BY order_date DESC;",
            ),
            (
                "How many orders per customer segment?",
                "SELECT c.segment, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.segment;",
            ),
            (
                "Top 10 customers by total spending",
                "SELECT c.name, c.email, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name, c.email ORDER BY total_spent DESC LIMIT 10;",
            ),
        ]

        for i, (question, sql) in enumerate(examples, 1):
            schema_parts.append(f"\nExample {i}:")
            schema_parts.append(f"Question: {question}")
            schema_parts.append(f"SQL: {sql}")

        return "\n".join(schema_parts)

    def _static_schema_context(self) -> str:
        """
        Static fallback schema context used when the database is unreachable.
        Mirrors the tables created by scripts/init_db.py.
        """
        schema_parts = []
        schema_parts.append("DATABASE SCHEMA DOCUMENTATION")
        schema_parts.append("=" * 60)

        documentation = """
    This is an e-commerce database with three main tables:
    - customers: Contains customer information including name, email, segment (SMB, Enterprise, Individual), and country
    - products: Product catalog with name, category, price, stock quantity, and description
    - orders: Customer orders with order date, total amount, status (Pending, Delivered, Cancelled, Processing), and shipping address

    The customers table has a one-to-many relationship with orders (one customer can have many orders).

    IMPORTANT NOTES:
    - For order revenue/pricing, use orders.total_amount (NOT 'price')
    - Customer segments: 'SMB', 'Enterprise', 'Individual' (case-sensitive)
    - Order statuses: 'Pending', 'Delivered', 'Cancelled', 'Processing' (case-sensitive)
    - To join customers and orders: JOIN orders ON customers.id = orders.customer_id
    - Revenue: SUM(orders.total_amount)
    - Product categories include Electronics, Software, Hardware, Services
    """
        schema_parts.append(documentation)
        schema_parts.append("\nTABLE SCHEMAS:")
        schema_parts.append("-" * 60)

        schema_parts.append("""
    Table: customers
    Columns:
    - id (SERIAL PRIMARY KEY)
    - name (VARCHAR NOT NULL) - Customer full name
    - email (VARCHAR UNIQUE NOT NULL) - Customer email address
    - segment (VARCHAR NOT NULL) - One of: 'SMB', 'Enterprise', 'Individual'
    - country (VARCHAR NOT NULL) - Customer country of operation
    - created_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    - updated_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """)

        schema_parts.append("""
    Table: products
    Columns:
    - id (SERIAL PRIMARY KEY)
    - name (VARCHAR NOT NULL) - Product name
    - category (VARCHAR NOT NULL) - Product category (Electronics, Software, Hardware, Services)
    - price (DECIMAL NOT NULL, min 0.01) - Product unit price
    - stock_quantity (INT NOT NULL DEFAULT 0, min 0) - Current inventory count
    - description (TEXT) - Product description
    - created_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    - updated_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """)

        schema_parts.append("""
    Table: orders
    Columns:
    - id (SERIAL PRIMARY KEY)
    - customer_id (INT NOT NULL) - Foreign key to customers.id
    - order_date (DATE NOT NULL DEFAULT CURRENT_DATE) - Date of order
    - total_amount (DECIMAL NOT NULL, min 0) - TOTAL ORDER PRICE (use this for revenue, NOT 'price'!)
    - status (VARCHAR NOT NULL) - One of: 'Pending', 'Delivered', 'Cancelled', 'Processing'
    - shipping_address (TEXT) - Shipping address
    - created_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    - updated_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """)

        schema_parts.append("\nEXAMPLE QUERIES:")
        schema_parts.append("-" * 60)

        examples = [
            (
                "How many customers do we have?",
                "SELECT COUNT(*) as customer_count FROM customers;",
            ),
            (
                "What is the total revenue from all orders?",
                "SELECT SUM(total_amount) as total_revenue FROM orders;",
            ),
            (
                "List all delivered orders",
                "SELECT * FROM orders WHERE status = 'Delivered' ORDER BY order_date DESC;",
            ),
            (
                "How many orders per customer segment?",
                "SELECT c.segment, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.segment;",
            ),
            (
                "Top 10 customers by total spending",
                "SELECT c.name, c.email, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name, c.email ORDER BY total_spent DESC LIMIT 10;",
            ),
        ]

        for i, (question, sql) in enumerate(examples, 1):
            schema_parts.append(f"\nExample {i}:")
            schema_parts.append(f"Question: {question}")
            schema_parts.append(f"SQL: {sql}")

        return "\n".join(schema_parts)

    @track(name="text_to_sql_generate")
    async def generate_sql_for_approval(
        self,
        question: str,
        explain: bool = True,
        vanna_temperature: Optional[float] = None,
        vanna_seed: Optional[int] = None,
        vanna_top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        settings = get_settings()
        if not self.is_trained:
            self.complete_training()

        # Set dynamic overrides for this generation call
        if vanna_temperature is not None:
            self.vanna.current_temperature = vanna_temperature
        else:
            self.vanna.current_temperature = settings.llm_temperature

        if vanna_seed is not None:
            self.vanna.current_seed = vanna_seed
        else:
            self.vanna.current_seed = 42

        if vanna_top_p is not None:
            self.vanna.current_top_p = vanna_top_p
        else:
            self.vanna.current_top_p = 0.1

        if self.query_cache_service and (
            self.query_cache_service.enabled or self.query_cache_service.use_local
        ):
            cache_key = self.query_cache_service.get_sql_gen_key(question)
            cached_result = self.query_cache_service.get(
                cache_key, cache_type="sql_gen"
            )

            if cached_result and "sql" in cached_result:
                logger.info(
                    f"SQL generation cache HIT for question: '{question[:50]}...'"
                )
                return {
                    "query_id": str(uuid.uuid4()),
                    "question": question,
                    "sql": cached_result["sql"],
                    "explanation": (
                        cached_result.get(
                            "explanation",
                            "This SQL will retrieve data from your database. Please review before approving.",
                        )
                        if explain
                        else "Explanation omitted by request."
                    ),
                    "status": "pending_approval",
                    "generated_at": pd.Timestamp.now().isoformat(),
                    "cache_hit": True,
                    "cost_saved": "$0.08",
                }

        try:
            try:
                sql = await self.vanna.generate_sql_async(
                    question=question, schema_context=self.schema_context
                )
                explanation = (
                    "This SQL will retrieve data from your database. Please review before approving."
                    if explain
                    else "Explanation omitted by request."
                )
            except Exception as vanna_err:
                logger.warning(
                    f"Vanna SQL generation failed: {vanna_err}. Falling back to direct LLM SQL generation..."
                )
                from langchain_core.messages import HumanMessage
                from app.core.llm_factory import get_chat_llm

                llm = get_chat_llm()
                prompt = f"""
You are a senior database administrator.
Generate a PostgreSQL SQL query to answer the user question.
Use the schema context provided.

{self.schema_context}

Question: {question}

Respond strictly with the SQL query in a markdown code block starting with ```sql and ending with ```.
Do not include any additional text outside the code block.
"""
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                content = response.content
                # Extract SQL
                sql = None
                if "```sql" in content.lower():
                    parts = content.split("```")
                    for part in parts:
                        if part.strip().lower().startswith("sql"):
                            sql = part[3:].strip()
                if not sql:
                    sql = content.strip()

                logger.info(
                    f"Direct LLM SQL generation fallback succeeded! Generated SQL: {sql[:100]}..."
                )
                explanation = (
                    "⚠️ Direct LLM Fallback: Generated using the primary LLM (LiteLLM/Groq) as the core Vanna agent was unavailable."
                    if explain
                    else "Explanation omitted by request."
                )

            if self.query_cache_service and (
                self.query_cache_service.enabled or self.query_cache_service.use_local
            ):
                cache_key = self.query_cache_service.get_sql_gen_key(question)
                cache_value = {
                    "sql": sql,
                    "explanation": explanation,
                    "question": question,
                }
                ttl = settings.cache_ttl_sql_gen
                self.query_cache_service.set(
                    cache_key, cache_value, ttl=ttl, cache_type="sql_gen"
                )
                logger.info(
                    f"SQL generation cache MISS - cached for '{question[:50]}...' (TTL: {ttl}s)"
                )

            return {
                "query_id": str(uuid.uuid4()),
                "question": question,
                "sql": sql,
                "explanation": explanation,
                "status": "pending_approval",
                "cache_hit": False,
                "cost_saved": "$0.00",
            }
        except Exception as e:
            raise Exception(f"Failed to generate SQL: {str(e)}")

    def get_pending_queries(self) -> List[Dict[str, Any]]:
        """
        Deprecated: pending_queries are stored in the shared PendingStore
        (app.services.pending_store), not in TextToSQLService. This method
        is kept for backward compatibility — always returns empty list.
        """
        return []
